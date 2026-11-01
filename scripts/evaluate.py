"""Evaluate a trained checkpoint on one split.

Examples:
    # accuracy, macro-F1, kappa, ECE, NLL, Brier and AURC
    python scripts/evaluate.py --ckpt runs/d2_s0/best.pt --data data/d2/test.npz

    # noisy test at 20 dB
    python scripts/evaluate.py --ckpt runs/d2_s0/best.pt --data data/d2/test.npz --test-snr 20

    # out-of-distribution protocol: the checkpoint was trained with
    # --exclude-classes; events of the excluded classes are scored as unknown
    python scripts/evaluate.py --ckpt runs/d3_ood/best.pt --data data/d3/test.npz --ood --tiers 0.15 0.40
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.engine import dataset_from_config, evaluate_predictions, load_checkpoint, make_loader, predict  # noqa: E402
from daefnet.metrics import ood_auroc, tier_report  # noqa: E402


def parse_ops(fail, noise):
    ops = {}
    for m in fail or []:
        ops[int(m)] = "fail"
    for item in noise or []:
        m, snr = item.split(":")
        ops[int(m)] = f"noise:{snr}"
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="path to a split file, e.g. data/d2/test.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test-snr", type=float, default=None)
    ap.add_argument("--fail", nargs="*", default=None, help="sensor indices replaced by a flat zero signal")
    ap.add_argument("--noise", nargs="*", default=None, help="sensor:SNR pairs, e.g. 2:5")
    ap.add_argument("--ood", action="store_true", help="score classes absent from the checkpoint as unknown")
    ap.add_argument("--tiers", nargs=2, type=float, default=None, help="thresholds of the three-tier policy")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    cfg = ckpt["config"]
    classes = ckpt["classes"]
    ops = parse_ops(args.fail, args.noise)

    data = dataset_from_config(args.data, cfg, test_snr=args.test_snr, sensor_ops=ops, classes_keep=classes)
    pred = predict(model, make_loader(data, args.batch_size, False, args.num_workers), device)
    res = evaluate_predictions(pred)
    correct = pred["prob"].argmax(1) == pred["y"]
    res["mean_reliability"] = pred["rho"].mean(axis=0).tolist()
    if args.tiers:
        res["tiers"] = tier_report(pred["u"], correct, *args.tiers)

    if args.ood:
        probe = dataset_from_config(args.data, cfg)
        unknown = [c for c in probe.classes if c not in classes]
        if not unknown:
            raise SystemExit("--ood needs a checkpoint trained with --exclude-classes")
        ood = dataset_from_config(args.data, cfg, classes_keep=unknown)
        pred_ood = predict(model, make_loader(ood, args.batch_size, False, args.num_workers), device)
        res["ood_auroc"] = ood_auroc(pred["u"], pred_ood["u"])
        res["ood_events"] = int(len(pred_ood["u"]))
        if args.tiers:
            res["ood_tiers"] = tier_report(pred_ood["u"], None, *args.tiers)

    summary = {k: v for k, v in res.items() if k != "per_class_f1"}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
