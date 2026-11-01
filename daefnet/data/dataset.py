"""Multi-sensor dataset with on-the-fly tokenization.

Raw samples are kept in memory and tokenized in `__getitem__`, so noise
augmentation (Section 3.2) and the failure scenarios of Section 3.5 act on
the raw signal before the quality descriptors are computed.

Sensor operations (`sensor_ops`) map a sensor index to
    "fail"        flat zero signal
    "noise:<dB>"  additive white Gaussian noise at the given SNR
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import PHASOR_KIND, tokenize_sensor
from .io import load_events

KIND_IDS = {"voltage": 0, "current": 1, "phasor": 2, "voltage_current": 3}


def add_noise(x, snr_db, rng):
    power = np.mean(x ** 2, axis=-1, keepdims=True)
    std = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return x + rng.normal(size=x.shape) * std


class MultiSensorDataset(Dataset):
    def __init__(self, path, train=False, noise_aug=False, aug_prob=0.5, snr_range=(20.0, 50.0),
                 test_snr=None, sensor_ops=None, classes_keep=None, fraction=1.0, n_points=32,
                 max_order=25, seed=0):
        self.meta, y, self.sensors = load_events(path)
        self.f0 = float(self.meta["f0"])
        self.cycles = float(self.meta["window_cycles"])
        self.kinds = [s["kind"] for s in self.meta["sensors"]]
        self.classes = list(self.meta["classes"])
        self.train = train
        self.noise_aug = noise_aug
        self.aug_prob = aug_prob
        self.snr_range = snr_range
        self.test_snr = test_snr
        self.sensor_ops = dict(sensor_ops or {})
        self.n_points = n_points
        self.max_order = max_order
        self.seed = seed

        idx = np.arange(len(y))
        self.label_map = None
        if classes_keep is not None:
            keep_ids = [self.classes.index(c) for c in classes_keep]
            idx = idx[np.isin(y, keep_ids)]
            self.label_map = {old: new for new, old in enumerate(keep_ids)}
            self.classes = list(classes_keep)
        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            chosen = []
            for c in np.unique(y[idx]):
                members = idx[y[idx] == c]
                n_take = max(1, int(round(fraction * members.size)))
                chosen.append(rng.choice(members, n_take, replace=False))
            idx = np.sort(np.concatenate(chosen))
        self.index = idx
        self.y_all = y

    @property
    def num_sensors(self):
        return len(self.kinds)

    @property
    def num_classes(self):
        return len(self.classes)

    def labels(self):
        y = self.y_all[self.index]
        if self.label_map is not None:
            y = np.array([self.label_map[int(v)] for v in y])
        return y

    def distance(self):
        d = self.meta.get("distance")
        return None if d is None else np.asarray(d, dtype=np.float32)

    def specs(self):
        """Input dimensions of every sensor, read from the first event."""
        item = self[0]
        return [
            {
                "kind": KIND_IDS[k],
                "dt_in": int(s["tok_t"].shape[-1]),
                "df_in": int(s["tok_f"].shape[-1]),
                "n_spec": int(s["tok_f"].shape[0]),
            }
            for k, s in zip(self.kinds, item["sensors"])
        ]

    def __len__(self):
        return len(self.index)

    def _raw(self, m, i):
        s = self.sensors[m]
        x = np.array(s["x"][i], dtype=np.float64)
        fs = float(s["fs"][i])
        if s["t"] is not None:
            t = np.asarray(s["t"][i], dtype=np.float64)
        else:
            t = np.arange(x.shape[-1]) / fs
        return x, t, fs

    def __getitem__(self, k):
        i = int(self.index[k])
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed * 1000003 + i)
        aug_snr = None
        if self.train and self.noise_aug and rng.random() < self.aug_prob:
            aug_snr = rng.uniform(*self.snr_range)
        if self.test_snr is not None:
            aug_snr = self.test_snr

        out = []
        for m, kind in enumerate(self.kinds):
            x, t, fs = self._raw(m, i)
            if aug_snr is not None and kind != PHASOR_KIND:
                x = add_noise(x, aug_snr, rng)
            op = self.sensor_ops.get(m)
            if op == "fail":
                x = np.zeros_like(x)
            elif op is not None and op.startswith("noise:"):
                x = add_noise(x, float(op.split(":")[1]), rng)
            out.append(tokenize_sensor(x, t, fs, self.f0, kind, self.cycles, self.n_points, self.max_order))
        y = int(self.y_all[i])
        if self.label_map is not None:
            y = self.label_map[y]
        return {"sensors": out, "y": y}


def collate(batch):
    """Pad tokens of every sensor to the longest sequence in the batch."""
    n_sensors = len(batch[0]["sensors"])
    b = len(batch)
    sensors = []
    for m in range(n_sensors):
        items = [x["sensors"][m] for x in batch]
        n_max = max(it["tok_t"].shape[0] for it in items)
        dt = items[0]["tok_t"].shape[1]
        tok_t = np.zeros((b, n_max, dt), np.float32)
        tc = np.zeros((b, n_max), np.float32)
        mask_t = np.zeros((b, n_max), bool)
        for j, it in enumerate(items):
            n = it["tok_t"].shape[0]
            tok_t[j, :n] = it["tok_t"]
            tc[j, :n] = it["tc"]
            mask_t[j, :n] = True
        sensors.append(
            {
                "tok_t": torch.from_numpy(tok_t),
                "tc": torch.from_numpy(tc),
                "mask_t": torch.from_numpy(mask_t),
                "tok_f": torch.from_numpy(np.stack([it["tok_f"] for it in items])),
                "order_f": torch.from_numpy(np.stack([it["order_f"] for it in items])),
                "mask_f": torch.from_numpy(np.stack([it["mask_f"] for it in items])),
                "prior_on": torch.tensor([float(it["prior_on"]) for it in items]),
                "q": torch.from_numpy(np.stack([it["q"] for it in items])),
            }
        )
    y = torch.tensor([x["y"] for x in batch], dtype=torch.long)
    return {"sensors": sensors, "y": y}
