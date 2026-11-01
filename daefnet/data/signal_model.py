"""Parametric disturbance model of Eq. (2).

x(t) = A1(t) sin(w t + phi1) + sum_h A_h sin(h w t + phi_h) + x_tr(t) + eps(t)
A1(t) = A0 [1 +/- xi 1_[t1,t2](t)] [1 + zeta_f sin(w_f t)]

An event is described by a parameter dictionary drawn with `sample_event`
and evaluated at arbitrary time instants with `waveform`. Evaluating the
continuous model directly lets each emulated sensor sample the same event
at its own rate and clock offset (Eq. 3). Parameter ranges follow Table 4.
"""

import numpy as np

from .classes import components

TWO_PI = 2.0 * np.pi


def sample_event(label, rng, f0, window, freq=None, extended_harmonics=False):
    """Draw the parameters of one event.

    Args:
        label: class label, e.g. "sag+harmonics".
        rng: numpy Generator.
        f0: nominal frequency in Hz.
        window: recording window length in seconds; events start inside it.
        freq: actual system frequency in Hz (defaults to f0).
        extended_harmonics: add 11th and 13th harmonics (used for D4).
    """
    T = 1.0 / f0
    comps = components(label)
    p = {
        "f": float(f0 if freq is None else freq),
        "A0": rng.uniform(0.95, 1.05),
        "phi1": rng.uniform(0.0, TWO_PI),
    }

    rms_kind = [c for c in comps if c in ("sag", "swell", "interruption")]
    if rms_kind:
        kind = rms_kind[0]
        lo, hi = {"sag": (0.1, 0.9), "swell": (0.1, 0.8), "interruption": (0.9, 1.0)}[kind]
        xi = rng.uniform(lo, hi)
        sign = 1.0 if kind == "swell" else -1.0
        max_dur = min(9.0 * T, 0.9 * window)
        dur = rng.uniform(min(T, max_dur), max_dur)
        t1 = rng.uniform(0.0, window - dur)
        p["rms"] = (sign, xi, t1, t1 + dur)

    if "harmonics" in comps:
        harm = [(h, rng.uniform(0.05, 0.15), rng.uniform(0.0, TWO_PI)) for h in (3, 5, 7)]
        if extended_harmonics:
            harm += [(h, rng.uniform(0.0, 0.05), rng.uniform(0.0, TWO_PI)) for h in (11, 13)]
        p["harm"] = harm

    if "flicker" in comps:
        p["flicker"] = (rng.uniform(0.1, 0.2), rng.uniform(8.0, 25.0), rng.uniform(0.0, TWO_PI))

    if "impulsive" in comps:
        p["impulse"] = (
            rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0),  # signed peak (pu)
            rng.uniform(0.05e-3, 1.0e-3),  # duration (s)
            rng.uniform(0.1, 0.9) * window,  # centre time (s)
        )

    if "oscillatory" in comps:
        p["osc"] = (
            rng.uniform(0.1, 0.8),  # amplitude (pu)
            rng.uniform(300.0, 900.0),  # frequency (Hz)
            rng.uniform(8e-3, 40e-3),  # decay time constant (s)
            rng.uniform(0.1, 0.8) * window,  # onset (s)
            rng.uniform(0.0, TWO_PI),  # phase
        )

    if "notch" in comps:
        p["notch"] = (
            rng.uniform(0.1, 0.4),  # depth (pu)
            rng.uniform(0.01, 0.05) * T,  # width (s)
            rng.uniform(0.0, np.pi),  # position within each half cycle (rad)
        )

    # Per-phase scaling of transient terms (used by the three-phase renderer).
    p["tr_scale"] = tuple(rng.uniform(0.5, 1.0, size=3))
    return p


def nominal(p):
    """Undisturbed counterpart of an event (same fundamental)."""
    return {"f": p["f"], "A0": p["A0"], "phi1": p["phi1"], "tr_scale": p.get("tr_scale", (1.0, 1.0, 1.0))}


def waveform(p, t, phase=0):
    """Evaluate the model of Eq. (2) at times `t` (seconds).

    `phase` selects the phase (0, 1, 2) of a balanced three-phase set; it
    shifts the fundamental by -120 degrees per phase and scales transients.
    """
    t = np.asarray(t, dtype=np.float64)
    w = TWO_PI * p["f"]
    shift = -TWO_PI / 3.0 * phase

    env = np.full_like(t, p["A0"])
    if "rms" in p and phase in p.get("rms_phases", (0, 1, 2)):
        sign, xi, t1, t2 = p["rms"]
        env = env * (1.0 + sign * xi * ((t >= t1) & (t < t2)))
    if "flicker" in p:
        zeta, ff, phf = p["flicker"]
        env = env * (1.0 + zeta * np.sin(TWO_PI * ff * t + phf))

    theta = w * t + p["phi1"] + shift
    x = env * np.sin(theta)

    for h, a, ph in p.get("harm", ()):
        x = x + a * np.sin(h * (w * t + shift) + ph)

    scale = p.get("tr_scale", (1.0, 1.0, 1.0))[phase]

    if "impulse" in p:
        peak, dur, tc = p["impulse"]
        x = x + scale * peak * np.exp(-4.0 * np.log(2.0) * ((t - tc) / dur) ** 2)

    if "osc" in p:
        amp, fr, tau, t0, ph = p["osc"]
        on = t >= t0
        dt = np.where(on, t - t0, 0.0)
        x = x + scale * on * amp * np.exp(-dt / tau) * np.sin(TWO_PI * fr * dt + ph)

    if "notch" in p:
        depth, width, pos = p["notch"]
        half = np.mod(theta, np.pi)
        half_width = 0.5 * w * width
        inside = np.abs(half - pos) < half_width
        x = x - depth * inside * np.sign(np.sin(theta))

    return x


def three_phase(p, t):
    """Evaluate a balanced three-phase version of the event, shape (3, L)."""
    return np.stack([waveform(p, t, phase=k) for k in range(3)], axis=0)
