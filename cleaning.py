"""Lecture-style CLEAN: iterative, weighted time-domain sine-wave fitting.

This implements the prewhitening recipe in PHYS4143 Lecture 5: find a peak,
fit its sine/cosine coefficients, subtract the oscillation in the time
domain, and repeat. It is not image CLEAN or subtraction of power values.
The subtraction gain is one. Optional joint least-squares refitting reduces
cross-talk between extracted components; it does not resolve alias ambiguity.
All frequencies are cyclic microhertz and all actual timestamps are retained.
"""

from __future__ import annotations

import numpy as np

from time_series import _prepare

__all__ = ["clean_series", "reconstruct_components"]
_SCALE = 2 * np.pi * 1e-6


def _design(t, frequencies, intercept):
    columns = [np.ones_like(t)] if intercept else []
    for frequency in frequencies:
        phase = _SCALE * frequency * t
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns) if columns else np.empty((len(t), 0))


def _joint_fit(t, y, w, frequencies, intercept):
    design = _design(t, frequencies, intercept)
    if not design.shape[1]:
        return 0.0, np.zeros_like(y), y.copy(), np.empty(0), True
    weighted_design = design * np.sqrt(w[:, None])
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, y * np.sqrt(w), rcond=None)
    valid = rank == design.shape[1]
    baseline = float(coefficients[0]) if intercept else 0.0
    harmonic_coefficients = coefficients[int(intercept):]
    model = design[:, int(intercept):] @ harmonic_coefficients
    residual = y - baseline - model
    return baseline, model, residual, harmonic_coefficients, valid


def _single_fit(t, residual, w, frequency):
    phase = _SCALE * frequency * t
    s, c = np.sin(phase), np.cos(phase)
    ss, cc, sc = np.dot(w, s*s), np.dot(w, c*c), np.dot(w, s*c)
    det = ss*cc-sc*sc
    if ss <= 1e-12 or cc <= 1e-12 or det <= 1e-12*ss*cc:
        return np.nan, np.nan
    ys, yc = np.dot(w*residual, s), np.dot(w*residual, c)
    return (ys*cc-yc*sc)/det, (yc*ss-ys*sc)/det


class _Scanner:
    """Cache geometry once; cache full double-precision bases only if they fit."""

    def __init__(self, t, frequencies, w, cache_bytes):
        self.t, self.f, self.w = t, frequencies, w
        self.ss = np.full(len(frequencies), np.nan)
        self.cc = np.full(len(frequencies), np.nan)
        self.sc = np.full(len(frequencies), np.nan)
        self.sine = self.cosine = None
        needed = 2 * len(t) * len(frequencies) * 8
        if needed <= cache_bytes:
            phase = _SCALE * frequencies[:, None] * t
            self.sine, self.cosine = np.sin(phase), np.cos(phase)
        # The temporary phase/sine/cosine block remains bounded even without
        # a full cache. The caller's cache budget is never a data approximation.
        self.chunk = max(1, min(len(frequencies), 32*1024**2 // (3*8*len(t))))

    def powers(self, residual):
        amplitude = np.full(len(self.f), np.nan)
        wr = self.w * residual
        for start in range(0, len(self.f), self.chunk):
            stop = min(start+self.chunk, len(self.f))
            if self.sine is None:
                phase = _SCALE * self.f[start:stop, None] * self.t
                sine, cosine = np.sin(phase), np.cos(phase)
            else:
                sine, cosine = self.sine[start:stop], self.cosine[start:stop]
            if np.isnan(self.ss[start]):
                self.ss[start:stop] = np.einsum('ij,j,ij->i', sine, self.w, sine)
                self.cc[start:stop] = np.einsum('ij,j,ij->i', cosine, self.w, cosine)
                self.sc[start:stop] = np.einsum('ij,j,ij->i', sine, self.w, cosine)
            ss, cc, sc = self.ss[start:stop], self.cc[start:stop], self.sc[start:stop]
            ys, yc = sine @ wr, cosine @ wr
            det = ss*cc-sc*sc
            valid = (ss > 1e-12) & (cc > 1e-12) & (det > 1e-12*ss*cc)
            a = (ys[valid]*cc[valid]-yc[valid]*sc[valid])/det[valid]
            b = (yc[valid]*ss[valid]-ys[valid]*sc[valid])/det[valid]
            block = amplitude[start:stop]
            block[valid] = np.hypot(a, b)
        return amplitude


def _golden_minimum(objective, left, right, tolerance=1e-8):
    """Bounded scalar search; also retain endpoints for boundary peaks."""
    if not right > left:
        return float(left)
    left0, right0 = float(left), float(right)
    ratio = (np.sqrt(5)-1)/2
    a, b = right-ratio*(right-left), left+ratio*(right-left)
    fa, fb = objective(a), objective(b)
    for _ in range(64):
        if right-left <= tolerance:
            break
        if fa < fb:
            right, b, fb = b, a, fa
            a = right-ratio*(right-left)
            fa = objective(a)
        else:
            left, a, fa = a, b, fb
            b = left+ratio*(right-left)
            fb = objective(b)
    candidates = [left0, right0, a, b, (left+right)/2]
    return float(min(candidates, key=objective))


def _projected_objective(t, y, w, other_frequencies, intercept):
    """Variable projection: eliminate other coefficients before refining f."""
    sqrtw = np.sqrt(w)
    design = _design(t, other_frequencies, intercept) * sqrtw[:, None]
    if design.shape[1]:
        q, _ = np.linalg.qr(design, mode='reduced')
        target = sqrtw*y - q @ (q.T @ (sqrtw*y))
    else:
        q, target = None, sqrtw*y

    def objective(frequency):
        phase = _SCALE * frequency * t
        u, v = sqrtw*np.sin(phase), sqrtw*np.cos(phase)
        if q is not None:
            u = u-q @ (q.T @ u)
            v = v-q @ (q.T @ v)
        ss, cc, sc = np.dot(u, u), np.dot(v, v), np.dot(u, v)
        det = ss*cc-sc*sc
        if ss <= 1e-12 or cc <= 1e-12 or det <= 1e-12*ss*cc:
            return np.inf
        ys, yc = np.dot(u, target), np.dot(v, target)
        a, b = (ys*cc-yc*sc)/det, (yc*ss-ys*sc)/det
        residual = target-a*u-b*v
        return float(np.dot(residual, residual))
    return objective


def clean_series(time, values, frequency_grid, weights=None, time_unit='s',
                 max_components=10, amplitude_threshold=0.0, refine=True,
                 joint_refit=True, min_separation_uhz=None,
                 cache_bytes=128*1024**2, center=True):
    """Iteratively extract the largest remaining harmonic in a search grid.

    The sorted, strictly increasing positive ``frequency_grid`` is in microHz.
    ``amplitude_threshold`` is an explicit stopping amplitude, not a false-alarm
    probability. Stop also at ``max_components`` or numerical exhaustion.
    Default minimum separation is 0.25/T, in microHz; set it explicitly when
    modelling closely spaced structure. Selection/refinement cannot guarantee
    that a selected peak is the physical frequency rather than its alias.

    With ``joint_refit=True``, all selected coefficients are refitted together;
    ``center=True`` includes a fitted intercept. Thus this extension uses a
    floating constant, unlike the required mean-subtracted two-column scan.
    With ``refine=True``, new frequencies and one coordinate sweep of earlier
    frequencies are refined within their original neighbouring grid points;
    a final sweep polishes the completed component set without another scan.
    Joint refinement is a local least-squares improvement, not a global search.
    ``joint_refit=False`` performs literal successive full-amplitude subtraction
    after one initial weighted-mean removal (or zero baseline if center=False).

    Return ``mean``, oscillation-only ``model``, ``residual``, final component
    dictionaries, iteration history and a stop reason. The exact accounting
    identity is ``values = mean + model + residual``. Coefficients/phases use
    time relative to the first input observation. Weighted RSS is calculated
    with weights normalized to sum to one, hence has squared-data units.
    """
    t, y, w = _prepare(time, values, weights, time_unit)
    f = np.asarray(frequency_grid, dtype=float)
    if (f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)) or
            np.any(f <= 0) or np.any(np.diff(f) <= 0)):
        raise ValueError('frequency_grid must be finite, positive and strictly increasing')
    if (isinstance(max_components, (bool, np.bool_)) or
            not isinstance(max_components, (int, np.integer)) or max_components < 0):
        raise ValueError('max_components must be a nonnegative integer')
    threshold = float(amplitude_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('amplitude_threshold must be finite and nonnegative')
    for value, name in [(refine, 'refine'), (joint_refit, 'joint_refit'), (center, 'center')]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(name+' must be Boolean')
    if not isinstance(cache_bytes, (int, np.integer)) or cache_bytes < 0:
        raise ValueError('cache_bytes must be a nonnegative integer')
    span = np.ptp(t[w > 0])
    separation = 0.25e6/span if min_separation_uhz is None else float(min_separation_uhz)
    if not np.isfinite(separation) or separation < 0:
        raise ValueError('min_separation_uhz must be finite and nonnegative')

    baseline = float(np.dot(w, y)) if center else 0.0
    model = np.zeros_like(y)
    residual = y-baseline
    initial_rss = float(np.dot(w, residual**2))
    selected, bounds, final_coefficients, history = [], [], np.empty(0), []
    stop_reason = 'max_components'
    scanner = _Scanner(t, f, w, cache_bytes) if max_components else None
    for iteration in range(max_components):
        amplitudes = scanner.powers(residual)
        allowed = np.isfinite(amplitudes)
        for old in selected:
            allowed &= np.abs(f-old) >= max(separation, 1e-10)
        if not np.any(allowed):
            stop_reason = 'exhausted_frequencies'
            break
        candidate = int(np.argmax(np.where(allowed, amplitudes, -np.inf)))
        if amplitudes[candidate] < threshold:
            stop_reason = 'amplitude_threshold'
            break
        frequency = float(f[candidate])
        lo, hi = float(f[max(0, candidate-1)]), float(f[min(len(f)-1, candidate+1)])
        for old in selected:
            if old < frequency:
                lo = max(lo, old+separation)
            elif old > frequency:
                hi = min(hi, old-separation)
        if refine and hi > lo:
            target = y if joint_refit else residual
            others = selected if joint_refit else []
            objective = _projected_objective(t, target, w, others, center and joint_refit)
            refined = _golden_minimum(objective, lo, hi)
            if objective(refined) <= objective(frequency):
                frequency = refined
        a, b = _single_fit(t, residual, w, frequency)
        amplitude_before = float(np.hypot(a, b))
        if not np.isfinite(amplitude_before):
            stop_reason = 'rank_deficient'
            break
        if amplitude_before < threshold:
            stop_reason = 'amplitude_threshold'
            break
        trial_selected = selected+[frequency]
        trial_bounds = bounds+[(lo, hi)]
        if joint_refit:
            # One coordinate sweep after adding a component updates previous
            # frequencies that were initially biased by overlapping signals.
            if refine and len(trial_selected) > 1:
                for index in range(len(trial_selected)):
                    other = trial_selected[:index]+trial_selected[index+1:]
                    lower, upper = trial_bounds[index]
                    old_frequency = trial_selected[index]
                    for neighbour in other:
                        if neighbour < old_frequency:
                            lower = max(lower, neighbour+separation)
                        elif neighbour > old_frequency:
                            upper = min(upper, neighbour-separation)
                    if upper <= lower:
                        continue
                    objective = _projected_objective(t, y, w, other, center)
                    refined = _golden_minimum(objective, lower, upper)
                    if objective(refined) <= objective(old_frequency):
                        trial_selected[index] = refined
            trial_mean, trial_model, trial_residual, coefficients, valid = _joint_fit(
                t, y, w, trial_selected, center)
            if not valid:
                stop_reason = 'rank_deficient'
                break
        else:
            phase = _SCALE*frequency*t
            trial_mean, trial_model = baseline, model+a*np.sin(phase)+b*np.cos(phase)
            trial_residual = y-trial_mean-trial_model
            coefficients = np.r_[final_coefficients, a, b]
        rss = float(np.dot(w, trial_residual**2))
        previous_rss = float(np.dot(w, residual**2))
        if rss > previous_rss+1e-12*max(initial_rss, 1e-30):
            stop_reason = 'no_improvement'
            break
        selected, bounds = trial_selected, trial_bounds
        baseline, model, residual = trial_mean, trial_model, trial_residual
        final_coefficients = coefficients
        history.append({'iteration': iteration+1,
                        'selected_frequency_uhz': float(selected[-1]),
                        'amplitude_before_refit': amplitude_before,
                        'weighted_rss': rss, 'weighted_rms': float(np.sqrt(rss))})
    # A final coordinate sweep reduces the remaining cross-talk after the
    # last extraction. Its final RSS replaces the last history entry, so the
    # history describes the actually returned model and remains monotonic.
    if joint_refit and refine and len(selected) > 1:
        polished = selected.copy()
        for index in range(len(polished)):
            other = polished[:index]+polished[index+1:]
            lower, upper = bounds[index]
            old_frequency = polished[index]
            for neighbour in other:
                if neighbour < old_frequency:
                    lower = max(lower, neighbour+separation)
                elif neighbour > old_frequency:
                    upper = min(upper, neighbour-separation)
            if upper <= lower:
                continue
            objective = _projected_objective(t, y, w, other, center)
            refined = _golden_minimum(objective, lower, upper)
            if objective(refined) <= objective(old_frequency):
                polished[index] = refined
        trial_mean, trial_model, trial_residual, coefficients, valid = _joint_fit(
            t, y, w, polished, center)
        polished_rss = float(np.dot(w, trial_residual**2))
        if valid and polished_rss <= float(np.dot(w, residual**2)):
            selected = polished
            baseline, model, residual = trial_mean, trial_model, trial_residual
            final_coefficients = coefficients
            history[-1]['selected_frequency_uhz'] = float(selected[-1])
            history[-1]['weighted_rss'] = polished_rss
            history[-1]['weighted_rms'] = float(np.sqrt(polished_rss))
    components = []
    for i, frequency in enumerate(selected):
        a, b = map(float, final_coefficients[2*i:2*i+2])
        components.append({'frequency_uhz': float(frequency), 'alpha': a, 'beta': b,
                           'amplitude': float(np.hypot(a, b)), 'power': a*a+b*b,
                           'phase_rad': float(np.arctan2(b, a))})
    return {'mean': float(baseline), 'model': model, 'residual': residual,
            'components': components, 'history': history, 'stop_reason': stop_reason,
            'frequency_grid_uhz': f.copy(), 'initial_weighted_rss': initial_rss,
            'final_weighted_rss': float(np.dot(w, residual**2)),
            'min_separation_uhz': separation, 'joint_refit': bool(joint_refit),
            'reference_time': float(np.asarray(time)[0]), 'time_unit': time_unit}


def reconstruct_components(time, components, time_unit='s', reference_time=None):
    """Evaluate an oscillation-only component sum, keeping its original epoch."""
    units = {'s': 1.0, 'min': 60.0, 'h': 3600.0, 'd': 86400.0}
    times = np.asarray(time, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('time must be a nonempty, finite one-dimensional array')
    if time_unit not in units:
        raise ValueError("time_unit must be 's', 'min', 'h', or 'd'")
    epoch = times[0] if reference_time is None else float(reference_time)
    if not np.isfinite(epoch):
        raise ValueError('reference_time must be finite')
    t = (times-epoch)*units[time_unit]
    model = np.zeros_like(t)
    for component in components:
        f, a, b = (float(component[key]) for key in ['frequency_uhz', 'alpha', 'beta'])
        if not np.isfinite([f, a, b]).all() or f <= 0:
            raise ValueError('components must have finite coefficients and positive frequencies')
        phase = _SCALE*f*t
        model += a*np.sin(phase)+b*np.cos(phase)
    return model
