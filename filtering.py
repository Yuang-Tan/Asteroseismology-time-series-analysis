"""Time-domain Gaussian high-pass filtering at the supplied observation times.

The low-pass estimate uses a normalized Gaussian kernel and trapezoidal
integration weights. Subtracting it gives a symmetric, noncausal high-pass
filter suitable for retrospective analysis. No interpolation or FFT is used.
The continuous, infinitely supported amplitude response is
H(f) = 1 - exp(-2*pi**2*sigma**2*f**2). Cutoffs are the high-pass -3 dB
amplitude points, not the low-pass Gaussian's half-width.
"""

from __future__ import annotations

import numpy as np

_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def gaussian_sigma_seconds(cutoff_uhz):
    """Return sigma for H(cutoff)=1/sqrt(2), with cutoff in microhertz."""
    cutoff = float(cutoff_uhz)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff_uhz must be finite and strictly positive")
    return float(np.sqrt(-2 * np.log(1 - 1 / np.sqrt(2))) /
                 (2 * np.pi * cutoff * 1e-6))


def highpass_amplitude_response(frequency_uhz, cutoff_uhz):
    """Continuous-kernel amplitude response; frequencies are in microhertz."""
    frequencies = np.asarray(frequency_uhz, dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("frequencies must be finite and nonnegative")
    sigma = gaussian_sigma_seconds(cutoff_uhz)
    return -np.expm1(-2 * np.pi**2 * sigma**2 * (frequencies * 1e-6)**2)


def gaussian_highpass(time, values, cutoff_uhz, *, time_unit="s", truncate=5.0):
    """Filter values without replacing their actual, strictly increasing times.

    The local low-pass value at t_i is sum(q_j K_ij y_j)/sum(q_j K_ij),
    where q_j are trapezoidal time-integration weights and
    K_ij=exp[-(t_j-t_i)^2/(2*sigma^2)] inside +/-truncate*sigma.

    Returns a dict with ``highpass``, ``lowpass``, ``interior`` (a Boolean
    mask excluding truncated observation-boundary kernels), ``sigma_s`` and
    ``half_support_s``. Edge values are still returned, but should not be
    assigned the infinite-record transfer function. Filtering irregular or
    gapped data is not strictly shift invariant; compare measured gains
    against the continuous response where sampling is sufficiently dense.

    This is an offline filter: it uses observations on both sides of each
    time. Inputs must have at least three distinct, increasing timestamps.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if (t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or len(t) < 3 or
            not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))):
        raise ValueError("time and values must be finite, matching 1-D arrays of length >=3")
    if time_unit not in _SECONDS:
        raise ValueError("time_unit must be 's', 'min', 'h', or 'd'")
    t = (t - t[0]) * _SECONDS[time_unit]
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be distinct and strictly increasing")
    truncate = float(truncate)
    if not np.isfinite(truncate) or truncate < 3:
        raise ValueError("truncate must be finite and >=3 Gaussian standard deviations")
    sigma = gaussian_sigma_seconds(cutoff_uhz)
    half_support = truncate * sigma
    q = np.empty_like(t)
    q[0], q[-1] = (t[1] - t[0]) / 2, (t[-1] - t[-2]) / 2
    q[1:-1] = (t[2:] - t[:-2]) / 2
    left = np.searchsorted(t, t - half_support, side="left")
    right = np.searchsorted(t, t + half_support, side="right")
    low = np.empty_like(y)
    for i, (a, b) in enumerate(zip(left, right)):
        kernel = np.exp(-0.5 * ((t[a:b] - t[i]) / sigma)**2) * q[a:b]
        low[i] = np.dot(kernel, y[a:b]) / kernel.sum()
    interior = (t >= half_support) & (t <= t[-1] - half_support)
    return {"highpass": y - low, "lowpass": low, "interior": interior,
            "sigma_s": sigma, "half_support_s": half_support}
