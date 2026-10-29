"""Extra idea 5: weighted CLEAN and a gap-aware noise check for 46 LMi.

The file's scatter column supplies relative weights. Its absolute calibration
is not stated by the filename, so treating its numbers as m/s standard errors
is an explicit hypothesis tested below, not an assumed fact.
"""

from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from filtering import gaussian_sigma_seconds
from time_series import spectrum


def segment_operators(time_seconds, cutoff_uhz=300.0, gap_seconds=1800.0,
                      truncate=5.0):
    """Return independent Gaussian high-pass matrices for contiguous segments.

    Gaps larger than gap_seconds split the record. Kernels never span a
    segment boundary; only rows at least truncate*sigma from both boundaries
    are retained for noise comparison. The quadrature normalization is exact
    for a constant input, including irregular local sampling.
    """
    t = np.asarray(time_seconds, dtype=float)
    if (t.ndim != 1 or len(t) < 3 or not np.all(np.isfinite(t)) or
            np.any(np.diff(t) <= 0)):
        raise ValueError('times must be finite, strictly increasing, and length >=3')
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ValueError('gap_seconds must be finite and positive')
    if not np.isfinite(truncate) or truncate < 3:
        raise ValueError('truncate must be finite and at least 3')
    sigma = gaussian_sigma_seconds(cutoff_uhz)
    support = truncate * sigma
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > gap_seconds) + 1, len(t)]
    blocks = []
    for number, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        tt = t[start:stop]
        n = len(tt)
        if n < 3:
            h = np.zeros((n, n))
            interior = np.zeros(n, dtype=bool)
        else:
            q = np.empty(n)
            q[0], q[-1] = (tt[1]-tt[0])/2, (tt[-1]-tt[-2])/2
            q[1:-1] = (tt[2:]-tt[:-2])/2
            distance = tt[:, None] - tt[None, :]
            low = np.exp(-0.5 * (distance / sigma)**2) * q[None, :]
            low[np.abs(distance) > support] = 0
            low /= low.sum(axis=1)[:, None]
            h = np.eye(n) - low
            interior = (tt >= tt[0] + support) & (tt <= tt[-1] - support)
        blocks.append({'number': number, 'start': int(start), 'stop': int(stop),
                       'operator': h, 'interior': interior})
    return blocks


def apply_segment_operators(values, blocks):
    """Apply the fixed linear filter to one or several data columns."""
    y = np.asarray(values, dtype=float)
    if y.ndim not in (1, 2) or not np.all(np.isfinite(y)):
        raise ValueError('values must be finite and one- or two-dimensional')
    if not blocks or blocks[-1]['stop'] != len(y):
        raise ValueError('operator blocks do not match the data length')
    out = np.empty_like(y)
    for block in blocks:
        a, b = block['start'], block['stop']
        out[a:b] = block['operator'] @ y[a:b]
    return out


def propagated_noise_variance(scatter, blocks, weighted_design_q=None):
    """Diagonal of H D H^T, optionally after a fixed WLS projection.

    D=diag(scatter**2). If Q is an orthonormal basis for D**(-1/2) X,
    the CLEAN residual covariance conditional on its selected frequencies is
    D**(1/2)(I-QQ^T)D**(1/2). Subtract the corresponding low-rank term
    after applying H. This does not account for frequency-selection search.
    """
    sigma = np.asarray(scatter, dtype=float)
    if (sigma.ndim != 1 or not np.all(np.isfinite(sigma)) or
            np.any(sigma <= 0) or not blocks or blocks[-1]['stop'] != len(sigma)):
        raise ValueError('scatter must contain positive finite values matching the blocks')
    q = None if weighted_design_q is None else np.asarray(weighted_design_q, dtype=float)
    if q is not None and (q.ndim != 2 or q.shape[0] != len(sigma)):
        raise ValueError('weighted design basis has the wrong shape')
    variance = np.empty_like(sigma)
    for block in blocks:
        a, b = block['start'], block['stop']
        h = block['operator']
        hs = h * sigma[a:b][None, :]
        v = np.sum(hs**2, axis=1)
        if q is not None:
            v -= np.sum((hs @ q[a:b])**2, axis=1)
        variance[a:b] = np.maximum(v, 0)
    return variance


def _table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _spectrum_csv(path, result):
    columns = list(result)
    np.savetxt(path, np.column_stack([result[k] for k in columns]), delimiter=',',
               header=','.join(columns), comments='')


def run_analysis(root=None, show=False):
    """Run weighted CLEAN, segment filtering and a conditional noise calibration check."""
    from cleaning import clean_series

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    started = time.perf_counter()
    out, figures = root/'results', root/'figures'
    out.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    entry = next(x for x in manifest['datasets'] if x['filename'].startswith('46LMi-'))
    source = root/'data/raw'/entry['filename']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
    data = np.loadtxt(source)
    time_day, velocity, scatter = data.T
    if np.any(scatter <= 0) or not np.all(np.isfinite(data)):
        raise ValueError('46 LMi data or scatter values are invalid')
    t = (time_day-time_day[0]) * 86400
    weights = 1 / scatter**2
    grid = np.arange(5.0, 120.001, 0.05)
    background_grid = np.arange(1000.0, 1800.001, 1.0)
    background = spectrum(time_day, velocity, background_grid, weights=weights, time_unit='d')
    noise_amplitude = float(np.mean(background['amplitude']))
    threshold = 4 * noise_amplitude
    clean = clean_series(time_day, velocity, grid, weights=weights, time_unit='d',
                         max_components=20, amplitude_threshold=threshold,
                         min_separation_uhz=0.5e6/np.ptp(t), refine=True, joint_refit=True)
    components = clean['components']
    residual = clean['residual']
    reconstruction_error = float(np.max(np.abs(velocity-clean['mean']-clean['model']-residual)))
    assert reconstruction_error < 1e-9
    history_rss = np.array([r['weighted_rss'] for r in clean['history']])
    assert np.all(np.diff(history_rss) <= 1e-8 * max(1, history_rss[0]))
    original_spectrum = spectrum(time_day, velocity, grid, weights=weights, time_unit='d')
    residual_spectrum = spectrum(time_day, residual, grid, weights=weights, time_unit='d', center=False)
    _spectrum_csv(out/'extra5_original_spectrum.csv', original_spectrum)
    _spectrum_csv(out/'extra5_clean_residual_spectrum.csv', residual_spectrum)
    _spectrum_csv(out/'extra5_background_spectrum.csv', background)
    _table(out/'extra5_clean_components.csv', components)
    _table(out/'extra5_clean_history.csv', clean['history'])

    # Model frequencies are now fixed. Correct the variance for their linear
    # WLS coefficient fit, but do not claim an exact post-selection test.
    columns = [np.ones(len(t))]
    for comp in components:
        phase = 2*np.pi*comp['frequency_uhz']*1e-6*t
        columns.extend([np.sin(phase), np.cos(phase)])
    design = np.column_stack(columns)
    q, rr = np.linalg.qr(design/scatter[:, None], mode='reduced')
    assert np.linalg.matrix_rank(rr) == rr.shape[0]
    cutoff, gap_seconds, truncate = 300.0, 1800.0, 5.0
    blocks = segment_operators(t, cutoff, gap_seconds, truncate)
    hp = apply_segment_operators(residual, blocks)
    raw_variance = propagated_noise_variance(scatter, blocks)
    conditional_variance = propagated_noise_variance(scatter, blocks, q)
    use = np.zeros(len(t), dtype=bool)
    segment_id = np.empty(len(t), dtype=int)
    for b in blocks:
        use[b['start']:b['stop']] = b['interior']
        segment_id[b['start']:b['stop']] = b['number']
    use &= conditional_variance > 0
    expected = np.sqrt(conditional_variance)
    z = hp[use] / expected[use]
    ratio = float(np.sqrt(np.mean(z*z)))
    robust = float(1.4826*np.median(np.abs(z-np.median(z))))
    assert use.sum() >= 100

    # Repeat the fixed linear operations on independent noise with the supplied
    # scatter scale. This checks both covariance propagation and filter-induced
    # correlations; it intentionally does not rerun adaptive frequency selection.
    rng = np.random.default_rng(20261029)
    draws = 256
    white = rng.standard_normal((len(t), draws))
    clean_noise = scatter[:, None] * (white-q @ (q.T @ white))
    filtered_noise = apply_segment_operators(clean_noise, blocks)
    mock_rms = np.sqrt(np.mean((filtered_noise[use]/expected[use, None])**2, axis=0))
    mock_interval = np.quantile(mock_rms, [.025, .975])
    segment_rows = []
    for b in blocks:
        idx = np.arange(b['start'], b['stop'])[use[b['start']:b['stop']]]
        if len(idx) < 8:
            continue
        segment_rows.append({'segment': b['number'], 'start_day': float(time_day[b['start']]),
                             'interior_points': len(idx),
                             'observed_hp_rms_m_per_s': float(np.sqrt(np.mean(hp[idx]**2))),
                             'predicted_hp_rms_if_scatter_m_per_s': float(np.sqrt(np.mean(conditional_variance[idx]))),
                             'median_listed_scatter': float(np.median(scatter[idx])),
                             'normalized_rms_if_scatter_m_per_s': float(np.sqrt(np.mean((hp[idx]/expected[idx])**2)))})
    _table(out/'extra5_segment_noise.csv', segment_rows)
    np.savetxt(out/'extra5_time_series.csv',
               np.column_stack([time_day, velocity, scatter, weights, clean['model'], residual,
                                hp, expected, np.sqrt(raw_variance), use.astype(int), segment_id]),
               delimiter=',', comments='',
               header='time_day,velocity_m_per_s,listed_scatter,relative_inverse_variance_weight,clean_model_m_per_s,clean_residual_m_per_s,highpass_residual_m_per_s,predicted_hp_sigma_if_scatter_m_per_s,predicted_hp_sigma_without_fit_correction,comparison_mask,segment')
    np.savetxt(out/'extra5_conditional_noise_mc.csv', mock_rms, delimiter=',',
               header='normalized_rms', comments='')
    observed_by_segment = np.array([r['observed_hp_rms_m_per_s'] for r in segment_rows])
    predicted_by_segment = np.array([r['predicted_hp_rms_if_scatter_m_per_s'] for r in segment_rows])
    correlation = float(np.corrcoef(observed_by_segment, predicted_by_segment)[0, 1])
    original_wrms = float(np.sqrt(np.average((velocity-np.average(velocity, weights=weights))**2, weights=weights)))
    clean_wrms = float(np.sqrt(np.average(residual**2, weights=weights)))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.3), layout='constrained')
    axes[0, 0].plot(grid, original_spectrum['amplitude'], label='Original', lw=.8)
    axes[0, 0].plot(grid, residual_spectrum['amplitude'], label='CLEAN residual', lw=.8)
    axes[0, 0].set(xlabel='Frequency (microHz)', ylabel='Weighted amplitude (m/s)',
                   title='(a) 46 LMi: 20-term CLEAN limit', xlim=(5, 120))
    axes[0, 0].legend(fontsize=8)
    iterations = [0] + [r['iteration'] for r in clean['history']]
    rms_history = [original_wrms] + [r['weighted_rms'] for r in clean['history']]
    axes[0, 1].plot(iterations, rms_history, marker='o', ms=3)
    axes[0, 1].set(xlabel='Selected terms', ylabel='Weighted residual RMS (m/s)',
                   title='(b) Iterative subtraction; aliases remain possible')
    # Show a densely populated retained segment, not a line across data gaps.
    selected = max(blocks, key=lambda b: int(use[b['start']:b['stop']].sum()))
    ii = np.arange(selected['start'], selected['stop'])
    ii = ii[use[ii]]
    hours = (t[ii]-t[ii[0]])/3600
    axes[1, 0].plot(hours, hp[ii], '.-', lw=.65, ms=3, label='Filtered residual')
    axes[1, 0].fill_between(hours, -3*expected[ii], 3*expected[ii], alpha=.25,
                           label='±3 predicted sigma if listed scatter is m/s')
    axes[1, 0].set(xlabel='Hours within one retained segment', ylabel='Velocity (m/s)',
                   title='(c) Scatter-scale hypothesis is too small')
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].scatter(predicted_by_segment, observed_by_segment, s=20, label='Segments (at least 8 points)')
    low = predicted_by_segment.min()*.8
    high = predicted_by_segment.max()*1.2
    xx = np.geomspace(low, high, 100)
    axes[1, 1].plot(xx, xx, ':', color='0.5', label='Equal absolute scale')
    xx_fit = np.geomspace(predicted_by_segment.min()*.85, predicted_by_segment.max()*1.15, 100)
    axes[1, 1].plot(xx_fit, ratio*xx_fit, '--', label=f'Common RMS scale ×{ratio:.1f}')
    axes[1, 1].set(xscale='log', yscale='log', xlabel='Predicted HP RMS if listed scatter is m/s',
                   ylabel='Observed HP RMS (m/s)', title='(d) Conditional scale and relative variation',
                   xlim=(low, high), ylim=(low*.8, observed_by_segment.max()*1.2))
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(figures/'10_46lmi_clean_noise.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    summary = {
        'extra_idea': 5, 'dataset': '46 LMi (SONG)', 'observations': len(t),
        'span_days': float(np.ptp(time_day)), 'weights': '1 / listed_scatter**2',
        'scatter_calibration_caveat': 'The filename does not specify the absolute calibration of the scatter column. Its use as m/s standard deviations is a tested hypothesis, not established metadata.',
        'median_listed_scatter': float(np.median(scatter)),
        'clean_frequency_grid_microHz': {'min': 5.0, 'max': 120.0, 'step': .05},
        'clean_components': len(components), 'clean_max_components': 20,
        'clean_stop_reason': clean['stop_reason'],
        'amplitude_threshold_m_per_s': threshold,
        'threshold_definition': 'Four times mean original weighted amplitude at 1000-1800 microHz; a descriptive stopping rule, not false-alarm probability.',
        'selected_frequencies_microHz': [c['frequency_uhz'] for c in components],
        'original_weighted_rms_m_per_s': original_wrms, 'clean_weighted_rms_m_per_s': clean_wrms,
        'maximum_reconstruction_error_m_per_s': reconstruction_error,
        'highpass_cutoff_microHz': cutoff, 'segment_gap_seconds': gap_seconds,
        'segments': len(blocks), 'kernel_sigma_seconds': gaussian_sigma_seconds(cutoff),
        'edge_trim_seconds_per_segment_end': truncate*gaussian_sigma_seconds(cutoff),
        'retained_noise_comparison_samples': int(use.sum()),
        'highpass_rms_m_per_s': float(np.sqrt(np.mean(hp[use]**2))),
        'predicted_hp_rms_if_scatter_m_per_s': float(np.sqrt(np.mean(conditional_variance[use]))),
        'normalized_mean_if_scatter_m_per_s': float(np.mean(z)),
        'normalized_rms_if_scatter_m_per_s': ratio,
        'normalized_mad_sigma_if_scatter_m_per_s': robust,
        'rms_calibration_m_per_s_per_listed_scatter_unit': ratio,
        'segment_observed_vs_predicted_rms_correlation': correlation,
        'conditional_mc_draws': draws, 'conditional_mc_rms_95_percent_interval': mock_interval.tolist(),
        'conditional_mc_mean_normalized_rms': float(mock_rms.mean()),
        'variance_fit_correction_median_fraction': float(np.median(1-conditional_variance[use]/raw_variance[use])),
        'absolute_scatter_scale_supported': bool(mock_interval[0] <= ratio <= mock_interval[1]),
        'interpretation': 'The listed scatter supplies relative weights, but its numerical values do not account for the observed high-pass noise if interpreted as absolute m/s standard errors. A common RMS scale is reported as calibration, not proof that the remaining scatter is exclusively instrumental.',
        'limitations': 'Noise propagation and Monte Carlo condition on the fitted CLEAN frequencies; adaptive frequency selection is not rerun. Residual stellar variability, correlations and unreported calibration can contribute. Twenty extracted terms are not twenty identified modes. No filter connects separate observing segments.',
        'elapsed_seconds': time.perf_counter()-started,
    }
    (out/'extra5_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
