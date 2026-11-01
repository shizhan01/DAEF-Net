"""Sensor emulation following Eq. (3).

x_c^(m)[n] = Q_b( (g_m * x_c)(n / f_s + delta_m) ) + eta_c^(m)[n]

A reference waveform is rendered at FS_REF, passed through the sensor
response g_m, sampled at the sensor rate with clock offset delta_m,
quantized to b bits and corrupted with measurement noise. Timestamps are
reported on the nominal grid, so the offset is not visible to the model.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter

FS_REF = 51200.0


@dataclass
class WaveformSensor:
    name: str
    fs: float
    bits: int = 16
    full_scale: float = 2.0
    noise_std: float = 1e-4
    antialias_hz: Optional[float] = None  # cutoff of a 2nd-order Butterworth filter
    kind: str = "voltage"


@dataclass
class PhasorSensor:
    name: str
    fps: float
    noise_std: float = 1e-4
    kind: str = "phasor"


def quantize(x, bits, full_scale):
    lsb = 2.0 * full_scale / (2 ** bits)
    return np.clip(np.round(x / lsb) * lsb, -full_scale, full_scale - lsb)


def observe_waveform(sensor, t_ref, x_ref, window, offset, rng):
    """Sample a reference waveform with one waveform sensor.

    Returns the samples, shape (1, L), and the nominal timestamps, shape (L,).
    """
    x = x_ref
    if sensor.antialias_hz is not None:
        b, a = butter(2, sensor.antialias_hz, fs=FS_REF)
        x = lfilter(b, a, x)
    n = int(round(window * sensor.fs))
    t_nom = np.arange(n) / sensor.fs
    xs = np.interp(t_nom + offset, t_ref, x)
    xs = quantize(xs, sensor.bits, sensor.full_scale)
    xs = xs + rng.normal(0.0, sensor.noise_std, size=n)
    return xs[None, :].astype(np.float32), t_nom


def observe_phasor(sensor, t_ref, x_ref, f0, window, offset, rng, points_per_cycle=64):
    """Estimate a phasor stream with a one-cycle DFT.

    Channels: magnitude (pu peak), angle relative to the first frame (rad),
    frequency deviation (Hz) and rate of change of frequency (Hz/s).
    Returns an array of shape (4, K) and nominal frame times of shape (K,).
    """
    T = 1.0 / f0
    k = int(round(window * sensor.fps))
    t_frames = (np.arange(k) + 1) / sensor.fps
    grid = (np.arange(points_per_cycle) - points_per_cycle) * (T / points_per_cycle)
    tt = (t_frames + offset)[:, None] + grid[None, :]
    xx = np.interp(tt, t_ref, x_ref)
    ph = (2.0 / points_per_cycle) * np.sum(xx * np.exp(-1j * 2.0 * np.pi * f0 * tt), axis=1)
    ph = ph + sensor.noise_std * (rng.normal(size=k) + 1j * rng.normal(size=k))
    mag = np.abs(ph)
    ang = np.unwrap(np.angle(ph))
    dt = 1.0 / sensor.fps
    freq = f0 + np.gradient(ang, dt) / (2.0 * np.pi)
    rocof = np.gradient(freq, dt)
    out = np.stack([mag, ang - ang[0], freq - f0, rocof], axis=0)
    return out.astype(np.float32), t_frames
