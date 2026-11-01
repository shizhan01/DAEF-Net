"""Generators for the emulated multi-sensor set (D2) and the multi-bus set (D4).

D2: every event is rendered from Eq. (2) and observed by three sensors
through Eq. (3): S1 power quality monitor (12.8 kHz, 16 bit), S2 meter-grade
channel (3.2 kHz, 12 bit, 2nd-order anti-aliasing filter) and S3 phasor
stream (100 frames/s). Clock offsets are drawn from U[-50, 50] us.

D4: a disturbance is injected at a random metered bus of a feeder,
propagated to the other metered buses with Z_nk(f) / Z_kk(f) and
superimposed on background three-phase voltage and current waveforms.
Backgrounds are synthetic unless real snapshots are supplied.
"""

import numpy as np

from .classes import D2_CLASSES, D4_CLASSES
from .network import DEMO_METERED_BUSES, Feeder
from .sensors import FS_REF, PhasorSensor, WaveformSensor, observe_phasor, observe_waveform, quantize
from .signal_model import nominal, sample_event, three_phase, waveform


def d2_sensors():
    return [
        WaveformSensor("S1", fs=12800.0, bits=16, full_scale=2.0, noise_std=1e-4),
        WaveformSensor("S2", fs=3200.0, bits=12, full_scale=2.0, noise_std=1e-3, antialias_hz=1400.0),
        PhasorSensor("S3", fps=100.0, noise_std=1e-4),
    ]


def generate_d2(n_per_class, seed, f0=50.0, cycles=10, classes=D2_CLASSES, offset_us=50.0,
                extra_offset=None, freq=None):
    """Generate one D2 split.

    Args:
        n_per_class: events per class.
        seed: random seed; use different seeds for train, validation and test.
        extra_offset: optional per-sensor constant clock offset in seconds,
            added at generation time (used for the asynchrony test).
        freq: actual system frequency in Hz (frequency-deviation test).
    Returns:
        meta, y, sensors in the format of `daefnet.data.io.save_events`.
    """
    rng = np.random.default_rng(seed)
    sensors = d2_sensors()
    window = cycles / f0
    margin = 2.0 / f0 + 0.01
    t_ref = np.arange(-margin, window + 0.01, 1.0 / FS_REF)
    store = [{"x": [], "fs": [], "t": None} for _ in sensors]
    y = []
    for ci, label in enumerate(classes):
        for _ in range(n_per_class):
            p = sample_event(label, rng, f0, window, freq=freq)
            x_ref = waveform(p, t_ref)
            for m, s in enumerate(sensors):
                delta = rng.uniform(-offset_us, offset_us) * 1e-6
                if extra_offset is not None:
                    delta += extra_offset[m]
                if isinstance(s, PhasorSensor):
                    xs, _ = observe_phasor(s, t_ref, x_ref, f0, window, delta, rng)
                    store[m]["fs"].append(s.fps)
                else:
                    xs, _ = observe_waveform(s, t_ref, x_ref, window, delta, rng)
                    store[m]["fs"].append(s.fs)
                store[m]["x"].append(xs)
            y.append(ci)
    meta = {
        "name": "D2",
        "f0": f0,
        "window_cycles": cycles,
        "classes": list(classes),
        "sensors": [{"name": s.name, "kind": s.kind} for s in sensors],
        "distance": None,
    }
    return meta, np.asarray(y), store


def _fit_fundamental(x, t, f0):
    a = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)], axis=1)
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return float(np.hypot(coef[0], coef[1])), float(np.arctan2(coef[1], coef[0]))


def _synthetic_background(rng, n_bus, t, f0, ylc):
    """Three-phase background voltages and load currents, shape (n_bus, 6, L)."""
    out = np.zeros((n_bus, 6, t.size))
    for n in range(n_bus):
        amp = rng.uniform(0.96, 1.01)
        ang = rng.uniform(-0.05, 0.0)
        h5, h7 = rng.uniform(0.003, 0.015), rng.uniform(0.002, 0.01)
        p5, p7 = rng.uniform(0, 2 * np.pi, 2)
        y1 = ylc[n]
        for k in range(3):
            sh = -2.0 * np.pi / 3.0 * k
            th = 2 * np.pi * f0 * t + ang + sh
            out[n, k] = amp * np.sin(th) + h5 * np.sin(5 * th + p5) + h7 * np.sin(7 * th + p7)
            # current of a parallel R-L load at fundamental and two harmonics
            out[n, 3 + k] = (amp * np.abs(y1[0]) * np.sin(th + np.angle(y1[0]))
                             + h5 * np.abs(y1[1]) * np.sin(5 * th + p5 + np.angle(y1[1]))
                             + h7 * np.abs(y1[2]) * np.sin(7 * th + p7 + np.angle(y1[2])))
    return out


def generate_d4(n_per_class, seed, feeder=None, metered=None, f0=60.0, fs=2500.0, cycles=12,
                classes=D4_CLASSES, background=None, offset_us=50.0, bits=16, noise_std=2e-4):
    """Generate one D4 split.

    Args:
        feeder: `Feeder`; defaults to `Feeder.demo()`.
        metered: metered bus indices; defaults to the demo selection.
        background: optional array (N_bg, M, 6, L) of real per-unit snapshots
            (three voltages then three currents per metered bus).
    """
    rng = np.random.default_rng(seed)
    feeder = Feeder.demo() if feeder is None else feeder
    metered = list(DEMO_METERED_BUSES if metered is None else metered)
    n_m = len(metered)
    n = int(round(cycles * fs / f0))
    n_ext = 2 * n
    nfft = int(2 ** np.ceil(np.log2(2 * n_ext)))
    t_ext = (np.arange(n_ext) - n) / fs
    t_win = t_ext[n:]
    window = n / fs
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    trans = feeder.transfer(freqs, f0, metered)  # (F, src, dst)
    yl = feeder.load_admittance(freqs, f0, metered)  # (F, M)
    ylc = feeder.load_admittance(np.array([f0, 5 * f0, 7 * f0]), f0, metered).T  # (M, 3)

    store = [{"x": [], "fs": [], "t": None} for _ in range(n_m)]
    y = []
    for ci, label in enumerate(classes):
        for _ in range(n_per_class):
            k = int(rng.integers(n_m))
            if background is not None:
                bg = background[int(rng.integers(len(background)))][:, :, :n].astype(float)
            else:
                bg = _synthetic_background(rng, n_m, t_win, f0, ylc)
            p = sample_event(label, rng, f0, window, extended_harmonics=True)
            a0, phi = _fit_fundamental(bg[k, 0], t_win, f0)
            p["A0"], p["phi1"] = a0, phi
            if "rms" in p:
                p["rms_phases"] = (int(rng.integers(3)),) if rng.random() < 0.5 else (0, 1, 2)
            dv = three_phase(p, t_ext) - three_phase(nominal(p), t_ext)
            spec = np.fft.rfft(dv, n=nfft, axis=-1)  # (3, F)
            dv_bus = spec[None, :, :] * trans[:, k, :].T[:, None, :]  # (M, 3, F)
            di_bus = dv_bus * yl.T[:, None, :]
            delta = rng.uniform(-offset_us, offset_us, size=n_m) * 1e-6
            rot = np.exp(1j * 2 * np.pi * freqs[None, :] * delta[:, None])[:, None, :]
            dv_t = np.fft.irfft(dv_bus * rot, n=nfft, axis=-1)[..., n:n_ext]
            di_t = np.fft.irfft(di_bus * rot, n=nfft, axis=-1)[..., n:n_ext]
            x = bg.copy()
            x[:, :3] += dv_t
            x[:, 3:] += di_t
            x = quantize(x, bits, 3.0) + rng.normal(0.0, noise_std, size=x.shape)
            for m in range(n_m):
                store[m]["x"].append(x[m].astype(np.float32))
                store[m]["fs"].append(fs)
            y.append(ci)

    meta = {
        "name": "D4",
        "f0": f0,
        "window_cycles": cycles,
        "classes": list(classes),
        "sensors": [{"name": f"bus{b}", "kind": "voltage_current"} for b in metered],
        "distance": feeder.electrical_distance(metered).tolist(),
    }
    return meta, np.asarray(y), store
