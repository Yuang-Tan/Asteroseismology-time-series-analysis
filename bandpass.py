"""Finite-band harmonic least-squares projection on actual observation times.

This is an offline linear band-pass construction, not an ideal infinite-record
brick-wall filter. Frequencies use natural Fourier spacing 1/T, where T is
N times the mean observed spacing. Joint fitting accounts for nonorthogonal
columns at irregular timestamps; no FFT, interpolation, or unnormalized sum
of independently fitted oversampled coefficients is used.
"""

import numpy as np

_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def harmonic_bandpass(time, values, band_uhz, *, time_unit="s", weights=None):
    """Return the oscillatory projection onto a finite positive-frequency band.

    The natural grid is k/T (integer k, globally aligned to zero), restricted
    to the requested inclusive band. A constant is fitted jointly, but is not
    included in ``model``. ``values = mean + model + residual`` exactly.
    Dense, near-regular data are the intended use; poorly sampled dictionaries
    raise ValueError when rank deficient. A finite observation window causes
    leakage and rounded transitions for off-grid and band-edge signals.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    band = np.asarray(band_uhz, dtype=float)
    if (t.ndim != 1 or y.ndim != 1 or t.shape != y.shape or len(t) < 4 or
            not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))):
        raise ValueError("time and values must be finite matching 1-D arrays with >=4 samples")
    if time_unit not in _SECONDS:
        raise ValueError("unsupported time unit")
    t = (t-t[0])*_SECONDS[time_unit]
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    if band.shape != (2,) or not np.all(np.isfinite(band)) or not 0 < band[0] < band[1]:
        raise ValueError("band_uhz must contain finite 0 < lower < upper")
    period = float((t[-1]-t[0])*len(t)/(len(t)-1))
    spacing = 1e6/period
    bins = np.arange(int(np.ceil(band[0]/spacing)), int(np.floor(band[1]/spacing))+1)
    if not len(bins) or 2*len(bins)+1 >= len(t):
        raise ValueError("band must contain natural-grid frequencies and leave residual degrees of freedom")
    frequency = bins*spacing
    if weights is None:
        w = np.ones(len(t))
    else:
        w = np.asarray(weights, dtype=float)
        if (w.shape != t.shape or not np.all(np.isfinite(w)) or np.any(w <= 0)):
            raise ValueError("weights must be finite, matching and strictly positive")
        w = w/np.max(w)
    phase = 2*np.pi*1e-6*t[:, None]*frequency[None, :]
    design = np.column_stack((np.ones(len(t)), np.sin(phase), np.cos(phase)))
    root_weight = np.sqrt(w)
    coefficients, _, rank, singular = np.linalg.lstsq(
        design*root_weight[:, None], y*root_weight, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("band-pass design is rank deficient")
    alpha = coefficients[1:1+len(frequency)]
    beta = coefficients[1+len(frequency):]
    model = design[:, 1:]@coefficients[1:]
    residual = y-coefficients[0]-model
    return {"model": model, "residual": residual, "mean": float(coefficients[0]),
            "frequency_uhz": frequency, "alpha": alpha, "beta": beta,
            "amplitude": np.hypot(alpha, beta), "requested_band_uhz": band,
            "grid_spacing_uhz": spacing, "period_s": period,
            "rank": int(rank), "condition_number": float(singular[0]/singular[-1])}
