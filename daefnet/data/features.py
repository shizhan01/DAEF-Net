"""Per-sensor tokenization and signal-quality descriptors.

Temporal tokens (Eq. 4): patches defined by duration, l_p = T/8 and
l_s = T/16 with T = 1/f0, linearly interpolated to P points per channel.
Phasor streams, whose reporting interval exceeds l_p, use two consecutive
frames per token. Each token carries its centre time in seconds, which the
model turns into the continuous-time encoding of Eq. (5).

Spectral tokens (Eqs. 6, 7): Hann-windowed DFT of the whole window, with
bins grouped into bands of width f0/2 centred on the half-order harmonic
grid p f0 / 2. Windows shorter than two cycles fall back to integer orders.
Phasor streams use the same grid on the modulation spectrum of the
trajectory, with the frame rate in place of the sampling rate.

Quality descriptors q (Section 2.4.2): SNR estimate, missing-sample ratio,
timestamp jitter and clipped fraction. A flat zero signal gives
q = [0, 1, jitter, 0], which the model reuses when it drops a sensor.
"""

import numpy as np

PHASOR_KIND = "phasor"
PHASOR_SCALE = np.array([1.0, 1.0, 0.2, 0.002])
PHASOR_CLIP = 5.0
SNR_CAP_DB = 60.0


def bins_per_band(cycles):
    """Upper bound on the number of DFT bins inside one band."""
    if cycles >= 2.0:
        return int(np.floor(cycles / 2.0 + 1e-6)) + 1
    return int(np.floor(cycles + 1e-6)) + 1


def band_grid(fs, n_samples, f0, max_order=25):
    """Band centres (Hz), half width (Hz) and harmonic orders of Eq. (7)."""
    cycles = n_samples * f0 / fs
    h = max(1, min(max_order, int(np.floor(fs / (2.0 * f0)))))
    if cycles >= 2.0:
        p = np.arange(1, 2 * h + 1)
        return p * f0 / 2.0, f0 / 4.0, p / 2.0
    p = np.arange(1, h + 1)
    return p * f0, f0 / 2.0, p.astype(float)


def hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def spectral_tokens(x, fs, f0, cycles, max_order=25):
    """Harmonic-band spectral tokens of one sensor.

    Args:
        x: array (C, L).
        fs: sampling (or frame) rate in Hz.
        f0: nominal frequency in Hz.
        cycles: nominal window length in cycles, fixes the bins per band.
    Returns:
        tokens (2*max_order, C*2*Q), orders (2*max_order,), mask (2*max_order,)
    """
    c, n = x.shape
    q = bins_per_band(cycles)
    n_max = 2 * max_order
    w = hann(n)
    spec = np.fft.rfft(x * w, axis=-1) * (2.0 / w.sum())
    df = fs / n
    centres, half, orders = band_grid(fs, n, f0, max_order)
    feats = np.zeros((n_max, c, 2, q), dtype=np.float32)
    for i, centre in enumerate(centres[:n_max]):
        lo = int(np.ceil((centre - half) / df - 1e-6))
        hi = int(np.floor((centre + half) / df + 1e-6))
        lo, hi = max(lo, 0), min(hi, spec.shape[1] - 1)
        if hi < lo:
            continue
        idx = np.arange(lo, hi + 1)[:q]
        feats[i, :, 0, : idx.size] = spec[:, idx].real
        feats[i, :, 1, : idx.size] = spec[:, idx].imag
    nb = min(len(centres), n_max)
    mask = np.zeros(n_max, dtype=bool)
    mask[:nb] = True
    order = np.zeros(n_max, dtype=np.float32)
    order[:nb] = orders[:nb]
    return feats.reshape(n_max, c * 2 * q), order, mask


def temporal_tokens_waveform(x, t, f0, n_points=32):
    """Duration-based patches (Eq. 4); returns tokens (N, C*P) and centres (N,)."""
    c = x.shape[0]
    period = 1.0 / f0
    lp, ls = period / 8.0, period / 16.0
    step = np.median(np.diff(t)) if t.size > 1 else period
    duration = t[-1] - t[0] + step
    n_tok = int(np.floor((duration - lp) / ls + 1e-6)) + 1
    starts = t[0] + np.arange(n_tok) * ls
    tq = starts[:, None] + (np.arange(n_points) + 0.5) * (lp / n_points)
    patches = np.stack([np.interp(tq.ravel(), t, x[k]).reshape(n_tok, n_points) for k in range(c)], axis=1)
    return patches.reshape(n_tok, c * n_points).astype(np.float32), (starts + lp / 2.0).astype(np.float32)


def temporal_tokens_phasor(x, t):
    """Tokens made of two consecutive frames; returns (K-1, 2C) and centres."""
    tokens = np.concatenate([x[:, :-1].T, x[:, 1:].T], axis=1)
    centres = 0.5 * (t[:-1] + t[1:])
    return tokens.astype(np.float32), centres.astype(np.float32)


def scale_phasor(x):
    scale = PHASOR_SCALE if x.shape[0] == PHASOR_SCALE.size else np.ones(x.shape[0])
    return np.clip(x * scale[:, None], -PHASOR_CLIP, PHASOR_CLIP)


def _constant_mask(x, min_run, tol=1e-12):
    """Samples that belong to runs of at least `min_run` identical values."""
    out = np.zeros_like(x, dtype=bool)
    for k in range(x.shape[0]):
        xc = x[k]
        change = np.r_[True, np.abs(np.diff(xc)) > tol]
        run_id = np.cumsum(change) - 1
        counts = np.bincount(run_id)
        out[k] = counts[run_id] >= min_run
    return out


def _snr_waveform(x, t, f0, max_harmonic=7):
    """SNR of the residual after a cycle-wise fit of the fundamental and its
    low-order harmonics, in dB."""
    spc = int(round(1.0 / (f0 * np.median(np.diff(t)))))
    hmax = max(1, min(max_harmonic, spc // 2 - 1))
    cyc = np.floor((t - t[0]) * f0 + 1e-9).astype(int)
    fit_pow, res_pow = 0.0, 0.0
    for k in np.unique(cyc):
        sel = cyc == k
        if sel.sum() < 2 * hmax + 2:
            continue
        tk = t[sel]
        cols = [np.ones_like(tk)]
        for h in range(1, hmax + 1):
            cols += [np.cos(2 * np.pi * h * f0 * tk), np.sin(2 * np.pi * h * f0 * tk)]
        a = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(a, x[:, sel].T, rcond=None)
        fit = a @ coef
        fit_pow += float(np.sum(fit ** 2))
        res_pow += float(np.sum((x[:, sel].T - fit) ** 2))
    if fit_pow < 1e-12:
        return 0.0
    return float(np.clip(10.0 * np.log10(fit_pow / (res_pow + 1e-12)), 0.0, SNR_CAP_DB))


def _snr_phasor(x):
    mag = x[0]
    if np.max(np.abs(mag)) < 1e-9 or mag.size < 4:
        return 0.0
    smooth = np.convolve(mag, np.ones(3) / 3.0, mode="same")
    res = (mag - smooth)[1:-1]
    snr = 10.0 * np.log10(np.mean(mag ** 2) / (np.mean(res ** 2) + 1e-12))
    return float(np.clip(snr, 0.0, SNR_CAP_DB))


def quality_descriptors(x, t, fs, f0, kind):
    """Return q = [SNR/60, missing ratio, timestamp jitter, clipped fraction].

    `x` may contain NaN for missing samples; it is not modified.
    """
    nan = np.isnan(x)
    xf = np.where(nan, 0.0, x)
    if kind == PHASOR_KIND:
        const = _constant_mask(xf[:1], min_run=3)
        missing = float(np.mean(nan[:1] | const))
        snr = _snr_phasor(xf)
    else:
        min_run = max(8, int(round(fs / (8.0 * f0))))
        const = _constant_mask(xf, min_run=min_run)
        missing = float(np.mean(nan | const))
        snr = _snr_waveform(xf, t, f0)
    if t.size > 2:
        dt = np.diff(t)
        jitter = float(np.clip(np.std(dt - 1.0 / fs) * fs, 0.0, 1.0))
    else:
        jitter = 0.0
    amax = float(np.max(np.abs(xf)))
    if amax < 1e-9:
        clipped = 0.0
    else:
        near = np.abs(xf) >= amax * (1.0 - 1e-3)
        flat = np.concatenate([np.zeros((xf.shape[0], 1), bool), np.abs(np.diff(xf, axis=1)) <= 1e-12], axis=1)
        clipped = float(np.mean(near & flat))
    return np.array([snr / SNR_CAP_DB, missing, jitter, clipped], dtype=np.float32)


def tokenize_sensor(x, t, fs, f0, kind, cycles, n_points=32, max_order=25):
    """Full per-sensor preprocessing used by the dataset."""
    q = quality_descriptors(x, t, fs, f0, kind)
    x = np.nan_to_num(x, nan=0.0)
    if kind == PHASOR_KIND:
        xs = scale_phasor(x)
        tok_t, tc = temporal_tokens_phasor(xs, t)
        xm = xs - xs.mean(axis=1, keepdims=True)
        tok_f, order, mask_f = spectral_tokens(xm, fs, f0, cycles, max_order)
        prior_on = 0.0
    else:
        tok_t, tc = temporal_tokens_waveform(x, t, f0, n_points)
        tok_f, order, mask_f = spectral_tokens(x, fs, f0, cycles, max_order)
        prior_on = 1.0
    return {
        "tok_t": tok_t,
        "tc": tc,
        "tok_f": tok_f,
        "order_f": order,
        "mask_f": mask_f,
        "prior_on": np.float32(prior_on),
        "q": q,
    }
