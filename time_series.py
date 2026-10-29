"""Required PHYS4143 time-series assignment: weighted harmonic spectra.

At each trial frequency, fit ``alpha*sin(omega*t) + beta*cos(omega*t)``
using the two-column Frandsen least-squares equations. Frequencies enter and
leave this module in microhertz; phases use omega = 2*pi*f*1e-6 rad/s.

This is an amplitude-squared spectrum, not a power spectral density:
``amplitude = hypot(alpha, beta)`` and ``power = amplitude**2``. At the
correct frequency, an isolated noise-free sinusoid has power A**2 when its
baseline is known and ``center=False``. Time zero is the first input time.

The assignment's recommended mean subtraction is enabled by default. It
uses the weighted mean when weights are supplied. This is not a simultaneous
fit of an intercept: for short or unevenly sampled records, subtracting the
sample mean can change the fitted amplitude. No interpolation, FFT, CLEAN,
filtering, or stochastic-mode fitting is performed here.

Only NumPy is required by this module. Run ``python time_series.py --help``
for the command-line interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np


__all__ = ["spectrum", "spectral_window"]

_SECONDS_PER_UNIT = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_WORKING_BYTES = 32 * 1024 * 1024
_RANK_TOL = 1e-12
_SPECTRUM_COLUMNS = ("frequency_uhz", "alpha", "beta", "amplitude", "power")


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def _prepare(time, values, weights, time_unit):
    times = _vector(time, "time")
    data = _vector(values, "values")
    if len(times) != len(data):
        raise ValueError("time and values must have the same length")
    if len(times) < 2:
        raise ValueError("at least two observations are required")
    if time_unit not in _SECONDS_PER_UNIT:
        raise ValueError("time_unit must be 's', 'min', 'h', or 'd'")
    # Subtract first, then convert: large absolute epochs do not enter phases.
    relative_seconds = (times - times[0]) * _SECONDS_PER_UNIT[time_unit]
    if not np.all(np.isfinite(relative_seconds)):
        raise ValueError("the time span is too large to represent in seconds")

    if weights is None:
        normalized_weights = np.full(len(data), 1.0 / len(data))
    else:
        raw_weights = _vector(weights, "weights")
        if len(raw_weights) != len(data):
            raise ValueError("weights must have the same length as time")
        if np.any(raw_weights < 0) or not np.any(raw_weights > 0):
            raise ValueError("weights must be nonnegative with a positive sum")
        # Normalize in two stages to avoid overflow and preserve scale invariance.
        normalized_weights = raw_weights / np.max(raw_weights)
        normalized_weights = normalized_weights / np.sum(normalized_weights)
    active = normalized_weights > 0
    if np.count_nonzero(active) < 2:
        raise ValueError("at least two observations must have positive weight")
    if np.ptp(relative_seconds[active]) == 0:
        raise ValueError("positive-weight observations must span distinct times")
    return relative_seconds, data, normalized_weights


def _frequencies(frequencies_uhz):
    frequencies = _vector(frequencies_uhz, "frequencies_uhz")
    if np.any(frequencies < 0):
        raise ValueError("frequencies_uhz must be nonnegative")
    return frequencies


def _fit(relative_seconds, values, frequencies, weights):
    alpha = np.full(len(frequencies), np.nan)
    beta = np.full(len(frequencies), np.nan)
    weighted_values = weights * values
    # Three frequency-by-observation arrays (phase, sine, cosine) dominate
    # memory. Work on a bounded number of frequencies rather than N_f * N_t.
    chunk_size = max(1, _WORKING_BYTES // (3 * 8 * len(relative_seconds)))
    for start in range(0, len(frequencies), chunk_size):
        stop = min(start + chunk_size, len(frequencies))
        phase = (2.0 * np.pi * 1e-6 * frequencies[start:stop, None]) * relative_seconds
        sine = np.sin(phase)
        cosine = np.cos(phase)
        s = sine @ weighted_values
        c = cosine @ weighted_values
        ss = np.einsum("ij,j,ij->i", sine, weights, sine)
        cc = np.einsum("ij,j,ij->i", cosine, weights, cosine)
        sc = np.einsum("ij,j,ij->i", sine, weights, cosine)
        determinant = ss * cc - sc * sc
        # Check both column norms and their angle. Exact zero/Nyquist fits
        # can lose a column; near-collinear columns are also unreliable.
        valid = (
            (ss > _RANK_TOL)
            & (cc > _RANK_TOL)
            & (determinant > _RANK_TOL * ss * cc)
            & np.isfinite(determinant)
        )
        block_alpha = alpha[start:stop]
        block_beta = beta[start:stop]
        block_alpha[valid] = (s[valid] * cc[valid] - c[valid] * sc[valid]) / determinant[valid]
        block_beta[valid] = (c[valid] * ss[valid] - s[valid] * sc[valid]) / determinant[valid]
    invalid = np.isnan(alpha) | np.isnan(beta)
    if np.any(invalid):
        warnings.warn(
            f"{np.count_nonzero(invalid)} trial frequency/frequencies have a "
            "singular or nearly singular sine/cosine fit (including zero "
            "frequency); returning NaN there.",
            RuntimeWarning,
            stacklevel=3,
        )
    amplitude = np.hypot(alpha, beta)
    return {
        "frequency_uhz": frequencies.copy(),
        "alpha": alpha,
        "beta": beta,
        "amplitude": amplitude,
        "power": amplitude**2,
    }


def spectrum(time, values, frequencies_uhz, weights=None, time_unit="s", center=True):
    """Calculate the weighted sine/cosine least-squares spectrum.

    Parameters
    ----------
    time, values : one-dimensional array-like
        Observation times and measured values. Times need not be evenly
        spaced; duplicates and unsorted times are allowed. At least two
        positive-weight observations must occur at different times.
    frequencies_uhz : one-dimensional array-like
        Nonnegative trial cyclic frequencies in microhertz. Output ordering
        matches input ordering. Zero and rank-deficient fits return NaN with
        a RuntimeWarning rather than misleading finite amplitudes.
    weights : array-like or None
        Relative nonnegative least-squares weights, normally 1/sigma**2.
        Weights are not standard deviations. Zero weights omit observations.
        A common scale factor has no effect. None means equal weights.
    time_unit : {'s', 'min', 'h', 'd'}
        Unit of the input times. Internally, use seconds relative to time[0].
    center : bool
        Subtract the weighted sample mean before fitting (default True).
        This is not a jointly fitted floating intercept; use False for
        a known zero baseline or a synthetic spectral-window signal.

    Returns
    -------
    dict of ndarray
        frequency_uhz, alpha, beta, amplitude, power. Alpha multiplies sine
        and beta multiplies cosine at times relative to time[0]. For the
        convention A*sin(omega*(t-time[0])+phi), phi=atan2(beta, alpha).
        Power is amplitude squared in units of values squared, not PSD.
    """
    relative_seconds, data, normalized_weights = _prepare(time, values, weights, time_unit)
    frequencies = _frequencies(frequencies_uhz)
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be True or False")
    if center:
        data = data - np.dot(normalized_weights, data)
    return _fit(relative_seconds, data, frequencies, normalized_weights)


def spectral_window(time, frequencies_uhz, reference_uhz, weights=None, time_unit="s"):
    """Calculate the assignment's phase-averaged spectral window.

    Sample a unit sine and a unit cosine at reference_uhz on the actual
    observation times. Fit each on the requested frequency grid, with the
    same weights and without sample-mean subtraction. Return
    power = 0.5 * (sine_power + cosine_power), and amplitude = sqrt(power).
    Thus power is one at a nonsingular reference frequency (to numerical
    precision). This phase average describes the sampling response; it is
    not an additional observation of the star.
    """
    reference = float(reference_uhz)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_uhz must be a finite positive frequency")
    times = _vector(time, "time")
    relative_seconds, _, normalized_weights = _prepare(
        times, np.zeros_like(times), weights, time_unit
    )
    frequencies = _frequencies(frequencies_uhz)
    phase = (2 * np.pi * reference * 1e-6) * relative_seconds
    sine_result = _fit(relative_seconds, np.sin(phase), frequencies, normalized_weights)
    cosine_result = _fit(relative_seconds, np.cos(phase), frequencies, normalized_weights)
    power = 0.5 * (sine_result["power"] + cosine_result["power"])
    return {
        "frequency_uhz": frequencies.copy(),
        "power": power,
        "amplitude": np.sqrt(power),
        "sine_power": sine_result["power"],
        "cosine_power": cosine_result["power"],
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute a Frandsen sine/cosine least-squares amplitude/power spectrum."
    )
    parser.add_argument("--input", type=Path, required=True, help="numeric whitespace or CSV table")
    parser.add_argument("--time-unit", choices=tuple(_SECONDS_PER_UNIT), default="s")
    parser.add_argument("--frequency-min", type=float, required=True, help="minimum frequency in microHz")
    parser.add_argument("--frequency-max", type=float, required=True, help="maximum frequency in microHz")
    parser.add_argument("--frequency-step", type=float, required=True, help="frequency step in microHz")
    parser.add_argument("--output", type=Path, required=True, help="output CSV file")
    parser.add_argument("--time-column", type=int, default=0, help="zero-based time column (default 0)")
    parser.add_argument("--value-column", type=int, default=1, help="zero-based value column (default 1)")
    parser.add_argument("--weight-column", type=int, help="zero-based relative-weight column (not sigma)")
    parser.add_argument("--skiprows", type=int, default=0, help="number of leading header rows to skip")
    parser.add_argument("--delimiter", help="input delimiter; default comma for .csv, whitespace otherwise")
    parser.add_argument("--no-center", action="store_true", help="do not subtract the weighted mean")
    args = parser.parse_args(argv)
    bounds = np.array([args.frequency_min, args.frequency_max, args.frequency_step])
    if not np.all(np.isfinite(bounds)) or args.frequency_min < 0 or args.frequency_max < args.frequency_min or args.frequency_step <= 0:
        parser.error("frequency bounds must be finite, 0 <= min <= max, and step > 0")
    columns = [args.time_column, args.value_column]
    if args.weight_column is not None:
        columns.append(args.weight_column)
    if min(columns) < 0 or args.skiprows < 0:
        parser.error("column indices and skiprows must be nonnegative")
    intervals = (args.frequency_max - args.frequency_min) / args.frequency_step
    if not np.isfinite(intervals) or intervals > 10_000_000:
        parser.error("frequency grid exceeds 10 million points; use a larger step")
    # Include max when it falls on the grid; never extend by a whole extra step.
    count = int(np.floor(intervals + 1e-10)) + 1
    frequencies = args.frequency_min + args.frequency_step * np.arange(count)
    delimiter = args.delimiter
    if delimiter is None and args.input.suffix.lower() == ".csv":
        delimiter = ","
    try:
        table = np.loadtxt(args.input, delimiter=delimiter, skiprows=args.skiprows, ndmin=2)
        if max(columns) >= table.shape[1]:
            raise ValueError(f"input has only {table.shape[1]} columns")
        result = spectrum(
            table[:, args.time_column], table[:, args.value_column], frequencies,
            weights=None if args.weight_column is None else table[:, args.weight_column],
            time_unit=args.time_unit, center=not args.no_center,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output,
            np.column_stack([result[key] for key in _SPECTRUM_COLUMNS]),
            delimiter=",", header=",".join(_SPECTRUM_COLUMNS), comments="",
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(frequencies)} trial frequencies to {args.output}")


if __name__ == "__main__":
    _main()
