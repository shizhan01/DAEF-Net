"""Storage format for multi-sensor event sets.

One split is one .npz file with the keys

    meta    JSON string with f0, window_cycles, classes, sensors and an
            optional electrical distance matrix between sensors
    y       (N,) integer labels
    x{m}    samples of sensor m, either a dense array (N, C_m, L_m) or an
            object array of N arrays (C_m, L_i) when lengths differ
    fs{m}   (N,) sampling rate (frame rate for phasor streams) in Hz
    t{m}    optional timestamps in seconds relative to the window start,
            dense (N, L_m) or object array; if absent, t = n / fs

Each sensor entry in meta has the fields "name" and "kind", where kind is
one of "voltage", "current", "voltage_current" or "phasor". Waveforms are
expected in per unit; phasor streams carry magnitude (pu), angle (rad),
frequency deviation (Hz) and ROCOF (Hz/s).
"""

import json

import numpy as np

SENSOR_KINDS = ("voltage", "current", "voltage_current", "phasor")


def _pack(items, dtype):
    shapes = {np.shape(a) for a in items}
    if len(shapes) == 1:
        return np.stack([np.asarray(a, dtype=dtype) for a in items], axis=0)
    out = np.empty(len(items), dtype=object)
    for i, a in enumerate(items):
        out[i] = np.asarray(a, dtype=dtype)
    return out


def save_events(path, meta, y, sensors):
    """Write one split.

    Args:
        meta: dict, see module docstring.
        y: sequence of labels.
        sensors: list with one dict per sensor holding "x" (list of arrays),
            "fs" (list of floats) and optionally "t" (list of arrays).
    """
    for s in meta["sensors"]:
        if s["kind"] not in SENSOR_KINDS:
            raise ValueError(f"unknown sensor kind {s['kind']}")
    arrays = {"meta": np.array(json.dumps(meta)), "y": np.asarray(y, dtype=np.int64)}
    for m, s in enumerate(sensors):
        arrays[f"x{m}"] = _pack(s["x"], np.float32)
        arrays[f"fs{m}"] = np.asarray(s["fs"], dtype=np.float64)
        if s.get("t") is not None:
            arrays[f"t{m}"] = _pack(s["t"], np.float64)
    np.savez(path, **arrays)


def load_events(path):
    """Read one split into memory; returns (meta, y, sensors)."""
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta"]))
        y = d["y"].astype(np.int64)
        sensors = []
        for m in range(len(meta["sensors"])):
            sensors.append(
                {
                    "x": d[f"x{m}"],
                    "fs": d[f"fs{m}"],
                    "t": d[f"t{m}"] if f"t{m}" in d.files else None,
                }
            )
    return meta, y, sensors
