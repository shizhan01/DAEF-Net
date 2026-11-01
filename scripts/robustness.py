"""Sensor failure and corruption scenarios of Section 3.5.

For every scenario the accuracy is averaged over all choices of the affected
sensor (one failure, one corrupted sensor) or sensor pair (two failures).
A failed sensor outputs a flat zero signal; a corrupted sensor receives
white noise at 5 dB.

Example:
    python scripts/robustness.py --ckpt runs/d4_s0/best.pt --data data/d4/test.npz --out runs/d4_s0/robust.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.engine import dataset_from_config, load_checkpoint, make_loader, predict  # noqa: E402


def accuracy(model, cfg, classes, path, ops, device, batch_size, workers):
    data = dataset_from_config(path, cfg, sensor_ops=ops, classes_keep=classes)
    pred = predict(model, make_loader(data, batch_size, False, workers), device)
    return float(np.mean(pred["prob"].argmax(1) == pred["y"])), pred["rho"].mean(axis=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corrupt-snr", type=float, default=5.0)
    ap.add_argument("--max-pairs", type=int, default=None, help="limit the number of failed pairs evaluated")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    cfg, classes = ckpt["config"], ckpt["classes"]
    m = len(ckpt["specs"])

    def run(ops):
        return accuracy(model, cfg, classes, args.data, ops, device, args.batch_size, args.num_workers)

    res = {"intact": run({})[0]}
    one_fail, corrupt, rho_fail = [], [], []
    for i in range(m):
        acc, rho = run({i: "fail"})
        one_fail.append(acc)
        rho_fail.append(rho[i])
        corrupt.append(run({i: f"noise:{args.corrupt_snr}"})[0])
        print(f"sensor {i}: failed {acc:.4f} (rho {rho[i]:.3f}), corrupted {corrupt[-1]:.4f}")
    res["one_failed_per_sensor"] = one_fail
    res["one_failed_mean"] = float(np.mean(one_fail))
    res["rho_of_failed_sensor"] = rho_fail
    res["one_corrupted_per_sensor"] = corrupt
    res["one_corrupted_mean"] = float(np.mean(corrupt))

    if m > 2:
        pairs = list(itertools.combinations(range(m), 2))
        if args.max_pairs is not None:
            pairs = pairs[: args.max_pairs]
        two = [run({i: "fail", j: "fail"})[0] for i, j in pairs]
        res["two_failed_mean"] = float(np.mean(two))
        res["two_failed_pairs"] = len(pairs)

    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
