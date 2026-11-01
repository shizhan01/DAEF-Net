"""Generate D2: 29 classes observed by three emulated sensors.

Default sizes give 840 / 120 / 240 events per class (24,360 / 3,480 / 6,960).
Train, validation and test use disjoint seeds.

Examples:
    python scripts/generate_d2.py --out data/d2
    # test set with a 5 ms clock offset on S2 and S3 (Figure 3)
    python scripts/generate_d2.py --out data/d2_offset5ms --splits test --clock-offset-ms 5 --offset-sensors 1 2
    # test set at 49.5 Hz
    python scripts/generate_d2.py --out data/d2_49p5 --splits test --freq 49.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.data.io import save_events  # noqa: E402
from daefnet.data.synthetic import generate_d2  # noqa: E402

SPLITS = {"train": (840, 0), "val": (120, 1), "test": (240, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the number of events per class")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--clock-offset-ms", type=float, default=0.0)
    ap.add_argument("--offset-sensors", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--freq", type=float, default=None, help="actual system frequency in Hz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    extra = None
    if args.clock_offset_ms:
        extra = [args.clock_offset_ms * 1e-3 if m in args.offset_sensors else 0.0 for m in range(3)]
    for split in args.splits:
        n, k = SPLITS[split]
        n = max(1, int(round(n * args.scale)))
        meta, y, sensors = generate_d2(n, seed=args.seed * 10 + k, extra_offset=extra, freq=args.freq)
        path = os.path.join(args.out, f"{split}.npz")
        save_events(path, meta, y, sensors)
        print(f"{split}: {len(y)} events -> {path}")


if __name__ == "__main__":
    main()
