"""Convert SEED-PQD-v1 (D1) into the event format.

Download SEED_PQD_v1.zip from https://zenodo.org/records/11843312 and unzip
it. The script accepts either the MATLAB file (1000 x 100 x 17) or the
folder with the 17 CSV files (1000 x 100 each). The stratified split is
70 / 10 / 20 % (11,900 / 1,700 / 3,400 signals).

Example:
    python scripts/prepare_d1.py --src SEED_PQD_v1 --out data/d1
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daefnet.data.classes import D1_CLASSES  # noqa: E402
from daefnet.data.io import save_events  # noqa: E402

FS, F0 = 5000.0, 50.0


def _norm(name):
    return re.sub(r"[^a-z]", "", name.lower())


def load_mat(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        arrays = [v for k, v in d.items() if not k.startswith("__") and np.ndim(v) == 3]
    except NotImplementedError:
        import h5py  # MATLAB v7.3 files
        with h5py.File(path, "r") as fh:
            arrays = [np.array(fh[k]).T for k in fh.keys() if np.ndim(fh[k]) == 3]
    if not arrays:
        raise SystemExit(f"no 3-D array found in {path}")
    data = np.asarray(arrays[0], dtype=np.float64)
    if data.shape[-1] != len(D1_CLASSES):
        raise SystemExit(f"unexpected shape {data.shape}")
    return [data[:, :, k] for k in range(len(D1_CLASSES))]


def load_csv_dir(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if len(files) != len(D1_CLASSES):
        raise SystemExit(f"expected {len(D1_CLASSES)} CSV files in {path}, found {len(files)}")
    keyed = {}
    for f in files:
        stem = _norm(os.path.splitext(os.path.basename(f))[0])
        match = max((c for c in D1_CLASSES if _norm(c) in stem), key=lambda c: len(_norm(c)), default=None)
        if match is None or match in keyed:
            raise SystemExit(f"cannot map {f} to a class; rename the files to the class names")
        keyed[match] = np.loadtxt(f, delimiter=",")
    return [keyed[c] for c in D1_CLASSES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MATLAB file or folder with the CSV files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.path.isfile(args.src):
        per_class = load_mat(args.src)
    else:
        mats = glob.glob(os.path.join(args.src, "**", "*.mat"), recursive=True)
        per_class = load_mat(mats[0]) if mats else load_csv_dir(args.src)

    rng = np.random.default_rng(args.seed)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for k, signals in enumerate(per_class):
        idx = rng.permutation(len(signals))
        n_tr, n_va = int(0.7 * len(idx)), int(0.1 * len(idx))
        parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
        for name, sel in parts.items():
            splits[name][0].extend(signals[i][None, :].astype(np.float32) for i in sel)
            splits[name][1].extend([k] * len(sel))

    meta = {
        "name": "D1",
        "f0": F0,
        "window_cycles": 1,
        "classes": list(D1_CLASSES),
        "sensors": [{"name": "voltage", "kind": "voltage"}],
        "distance": None,
    }
    os.makedirs(args.out, exist_ok=True)
    for name, (x, y) in splits.items():
        path = os.path.join(args.out, f"{name}.npz")
        save_events(path, meta, y, [{"x": x, "fs": [FS] * len(x)}])
        print(f"{name}: {len(y)} signals -> {path}")


if __name__ == "__main__":
    main()
