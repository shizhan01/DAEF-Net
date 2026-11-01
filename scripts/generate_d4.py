"""Generate a multi-bus data set in the style of D4.

Nine disturbance classes are injected at a random metered bus, propagated
through the feeder by harmonic-domain superposition and superimposed on
background waveforms. Default sizes give 1,000 / 125 / 375 events per class
(9,000 / 1,125 / 3,375).

With no feeder files the built-in ten-bus demo feeder and synthetic
backgrounds are used. To use the SoCal 28-bus circuit, export its branch
and load data to CSV (columns from,to,r,x and bus,g,b in per unit), list the
metered buses, and provide per-unit background snapshots for each split as
.npz files with an array "x" of shape (N, M, 6, L): three phase voltages
followed by three phase currents of the M metered buses, at 2.5 kHz.

Examples:
    python scripts/generate_d4.py --out data/d4
    python scripts/generate_d4.py --out data/d4_socal --branches branches.csv --loads loads.csv \
        --metered <indices of the eight metered buses> --background-dir backgrounds
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.data.io import save_events  # noqa: E402
from daefnet.data.network import Feeder  # noqa: E402
from daefnet.data.synthetic import generate_d4  # noqa: E402

SPLITS = {"train": (1000, 0), "val": (125, 1), "test": (375, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--branches", default=None)
    ap.add_argument("--loads", default=None)
    ap.add_argument("--slack", type=int, default=0)
    ap.add_argument("--metered", nargs="*", type=int, default=None)
    ap.add_argument("--background-dir", default=None, help="directory with train.npz, val.npz, test.npz backgrounds")
    ap.add_argument("--f0", type=float, default=60.0)
    ap.add_argument("--fs", type=float, default=2500.0)
    ap.add_argument("--cycles", type=int, default=12)
    args = ap.parse_args()

    feeder = None
    if args.branches:
        feeder = Feeder.from_csv(args.branches, args.loads, slack=args.slack)
        if not args.metered:
            raise SystemExit("--metered is required with a custom feeder")

    os.makedirs(args.out, exist_ok=True)
    for split in args.splits:
        n, k = SPLITS[split]
        n = max(1, int(round(n * args.scale)))
        background = None
        if args.background_dir:
            background = np.load(os.path.join(args.background_dir, f"{split}.npz"))["x"]
        meta, y, sensors = generate_d4(n, seed=args.seed * 10 + k, feeder=feeder, metered=args.metered,
                                       f0=args.f0, fs=args.fs, cycles=args.cycles, background=background)
        path = os.path.join(args.out, f"{split}.npz")
        save_events(path, meta, y, sensors)
        print(f"{split}: {len(y)} events -> {path}")


if __name__ == "__main__":
    main()
