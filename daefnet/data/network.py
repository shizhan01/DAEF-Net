"""Positive-sequence feeder model.

Provides the Thevenin electrical distance of Eq. (19),

    d_mn = |Z_mm + Z_nn - 2 Z_mn|,  Z = Y_red^{-1},

and the frequency-dependent voltage transfer Z_nk(f) / Z_kk(f) used to
propagate a disturbance injected at bus k to the other buses by
harmonic-domain superposition (Section 3.1, D4).

Branch impedances and load admittances are in per unit at the nominal
frequency. Branches are series R + jhX at harmonic order h, and loads are
parallel R-L elements, y_load(h) = g - j b / h. The slack bus is an ideal
source and is removed from the nodal admittance matrix.
"""

import csv

import numpy as np


class Feeder:
    def __init__(self, n_bus, branches, loads=None, slack=0):
        self.n_bus = int(n_bus)
        self.branches = np.asarray(branches, dtype=float).reshape(-1, 4)
        self.loads = np.zeros((self.n_bus, 2)) if loads is None else np.asarray(loads, dtype=float)
        self.slack = int(slack)
        self.keep = np.array([i for i in range(self.n_bus) if i != self.slack])
        self.pos = {int(b): i for i, b in enumerate(self.keep)}

    def admittance(self, h, with_loads=True):
        y = np.zeros((self.n_bus, self.n_bus), dtype=complex)
        for f, t, r, x in self.branches:
            f, t = int(f), int(t)
            yb = 1.0 / (r + 1j * h * x)
            y[f, f] += yb
            y[t, t] += yb
            y[f, t] -= yb
            y[t, f] -= yb
        if with_loads:
            g, b = self.loads[:, 0], self.loads[:, 1]
            y[np.diag_indices(self.n_bus)] += g - 1j * b / h
        return y

    def reduced_impedance(self, h, with_loads=True):
        y = self.admittance(h, with_loads)
        return np.linalg.inv(y[np.ix_(self.keep, self.keep)])

    def electrical_distance(self, buses):
        """Eq. (19) between the given buses, network impedances only."""
        z = self.reduced_impedance(1.0, with_loads=False)
        idx = [self.pos[int(b)] for b in buses]
        zs = z[np.ix_(idx, idx)]
        diag = np.diag(zs)
        return np.abs(diag[:, None] + diag[None, :] - 2.0 * zs)

    def transfer(self, freqs, f0, buses, h_min=0.05):
        """Voltage transfer ratios, shape (n_freq, n_src, n_dst), between the
        given buses: entry [f, k, n] = Z_nk(f) / Z_kk(f)."""
        idx = [self.pos[int(b)] for b in buses]
        out = np.zeros((len(freqs), len(idx), len(idx)), dtype=complex)
        for i, f in enumerate(freqs):
            h = max(abs(f) / f0, h_min)
            z = self.reduced_impedance(h)[np.ix_(idx, idx)]
            out[i] = (z / np.diag(z)[None, :]).T
        return out

    def load_admittance(self, freqs, f0, buses, h_min=0.05):
        """Normalized load admittance y_load(h) / |y_load(1)|, shape (n_freq, n_bus)."""
        g = self.loads[buses, 0]
        b = self.loads[buses, 1]
        h = np.maximum(np.abs(np.asarray(freqs)) / f0, h_min)[:, None]
        y = g[None, :] - 1j * b[None, :] / h
        y1 = np.abs(g - 1j * b)
        return y / np.maximum(y1, 1e-9)[None, :]

    @classmethod
    def from_csv(cls, branch_csv, load_csv=None, slack=0):
        """Read branches (columns: from,to,r,x) and loads (columns: bus,g,b)."""
        with open(branch_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        branches = [[int(r["from"]), int(r["to"]), float(r["r"]), float(r["x"])] for r in rows]
        n_bus = 1 + max(max(b[0], b[1]) for b in branches)
        loads = np.zeros((n_bus, 2))
        if load_csv is not None:
            with open(load_csv, newline="") as fh:
                for r in csv.DictReader(fh):
                    loads[int(r["bus"])] = [float(r["g"]), float(r["b"])]
        return cls(n_bus, branches, loads, slack)

    @classmethod
    def demo(cls):
        """Ten-bus radial feeder with two laterals; bus 0 is the source."""
        branches = [
            [0, 1, 0.010, 0.060],
            [1, 2, 0.030, 0.050],
            [2, 3, 0.040, 0.060],
            [3, 4, 0.050, 0.070],
            [4, 5, 0.040, 0.050],
            [3, 6, 0.060, 0.050],
            [6, 7, 0.050, 0.040],
            [2, 8, 0.050, 0.060],
            [8, 9, 0.060, 0.050],
        ]
        tan_phi = np.tan(np.arccos(0.9))
        g = np.array([0.0, 0.0, 0.35, 0.30, 0.25, 0.20, 0.30, 0.25, 0.30, 0.20])
        loads = np.stack([g, g * tan_phi], axis=1)
        return cls(10, branches, loads, slack=0)


DEMO_METERED_BUSES = (2, 3, 4, 5, 6, 7, 8, 9)
