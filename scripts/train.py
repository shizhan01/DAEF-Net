"""Train DAEF-Net on one data source.

Example:
    python scripts/train.py --config configs/d2.yaml --data data/d2 --out runs/d2_s0 --seed 0
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.config import load_config, save_config  # noqa: E402
from daefnet.engine import (build_model, cosine_with_warmup, dataset_from_config, evaluate_predictions,  # noqa: E402
                            load_checkpoint, make_loader, predict, save_checkpoint, set_seed,
                            train_one_epoch)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="directory with train.npz, val.npz, test.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as model.use_harmonic_prior=false")
    ap.add_argument("--exclude-classes", nargs="*", default=None,
                    help="classes withheld from training (out-of-distribution protocol)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training labels to use")
    ap.add_argument("--init", default=None, help="checkpoint to initialize from (fine-tuning)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    os.makedirs(args.out, exist_ok=True)
    save_config(cfg, os.path.join(args.out, "config.yaml"))
    set_seed(args.seed)
    device = torch.device(args.device)
    tcfg = cfg["train"]

    probe = dataset_from_config(os.path.join(args.data, "val.npz"), cfg)
    keep = None
    if args.exclude_classes:
        keep = [c for c in probe.classes if c not in set(args.exclude_classes)]

    train_set = dataset_from_config(os.path.join(args.data, "train.npz"), cfg, train=True, classes_keep=keep,
                                    fraction=args.train_frac, seed=args.seed)
    val_set = dataset_from_config(os.path.join(args.data, "val.npz"), cfg, classes_keep=keep, seed=args.seed)
    test_set = dataset_from_config(os.path.join(args.data, "test.npz"), cfg, classes_keep=keep, seed=args.seed)

    specs = train_set.specs()
    distance = train_set.distance()
    model = build_model(specs, train_set.num_classes, cfg, distance).to(device)
    if args.init:
        src, _ = load_checkpoint(args.init, device)
        own = model.state_dict()
        loaded = {k: v for k, v in src.state_dict().items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        print(f"initialized {len(loaded)}/{len(own)} tensors from {args.init}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"sensors={train_set.num_sensors} classes={train_set.num_classes} params={n_params / 1e6:.2f}M "
          f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = make_loader(train_set, tcfg["batch_size"], True, tcfg["num_workers"])
    val_loader = make_loader(val_set, tcfg["batch_size"], False, tcfg["num_workers"])
    test_loader = make_loader(test_set, tcfg["batch_size"], False, tcfg["num_workers"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  cosine_with_warmup(tcfg["warmup_epochs"], tcfg["epochs"]))

    best, best_epoch, wait = -1.0, -1, 0
    ckpt_path = os.path.join(args.out, "best.pt")
    with open(os.path.join(args.out, "log.csv"), "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "lr", "train_loss", "val_acc", "val_macro_f1", "val_ece", "seconds"])
        for epoch in range(tcfg["epochs"]):
            start = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, tcfg, epoch, device)
            scheduler.step()
            res = evaluate_predictions(predict(model, val_loader, device))
            elapsed = time.time() - start
            log.writerow([epoch, optimizer.param_groups[0]["lr"], f"{loss:.5f}", f"{res['accuracy']:.5f}",
                          f"{res['macro_f1']:.5f}", f"{res['ece']:.5f}", f"{elapsed:.1f}"])
            fh.flush()
            print(f"epoch {epoch:3d} loss {loss:.4f} val acc {res['accuracy']:.4f} "
                  f"mF1 {res['macro_f1']:.4f} ece {res['ece']:.4f} ({elapsed:.0f}s)")
            if res["macro_f1"] > best:
                best, best_epoch, wait = res["macro_f1"], epoch, 0
                save_checkpoint(ckpt_path, model, specs, train_set.classes, cfg, distance, epoch, best)
            else:
                wait += 1
                if wait >= tcfg["patience"]:
                    print(f"early stopping at epoch {epoch}")
                    break

    model, _ = load_checkpoint(ckpt_path, device)
    res = evaluate_predictions(predict(model, test_loader, device))
    res["best_epoch"] = best_epoch
    res["params"] = n_params
    with open(os.path.join(args.out, "test_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_class_f1"}, indent=2))


if __name__ == "__main__":
    main()
