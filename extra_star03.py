"""Extra idea 1: CLEAN the main star03 delta Scuti pulsations.

The lecture's CLEAN is iterative time-domain sine-wave subtraction. The
selected frequencies are jointly refitted after each iteration; this is
not an image-domain CLEAN algorithm or a formal mode-identification test.
"""

from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from cleaning import clean_series
from time_series import spectrum


def _spectrum_csv(path, result):
    keys = list(result)
    np.savetxt(path, np.column_stack([result[k] for k in keys]),
               delimiter=',', header=','.join(keys), comments='')


def _rows_csv(path, rows):
    if not rows:
        raise ValueError('Expected at least one CLEAN component')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(root=None, show=False):
    """Recompute and save star03 spectra, CLEAN components and residuals."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results, figures = root / 'results', root / 'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((root / 'data/source_manifest.json').read_text())
    entry = next(e for e in manifest['datasets'] if e['filename'].startswith('star03-'))
    data_path = root / 'data/raw' / entry['filename']
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == entry['sha256']
    data = np.loadtxt(data_path)
    time_days, flux = data.T
    relative_days = time_days - time_days[0]
    span_s = np.ptp(time_days) * 86400.0
    resolution = 1e6 / span_s

    # A wide diagnostic scan motivates the finer 50-600 microHz search.
    # The wide grid is not used to estimate final component frequencies.
    overview_grid = np.arange(1.0, 4000.01, 0.5)
    overview_raw = spectrum(time_days, flux, overview_grid, time_unit='d')
    background_band = (overview_grid >= 2000) & (overview_grid <= 4000)
    background = float(np.mean(overview_raw['amplitude'][background_band]))
    threshold = 4.0 * background
    step = resolution / 4.0
    search_grid = 50.0 + step * np.arange(int(np.floor(550.0 / step)) + 1)
    raw = spectrum(time_days, flux, search_grid, time_unit='d')
    cleaned = clean_series(time_days, flux, search_grid, time_unit='d',
                           max_components=20, amplitude_threshold=threshold,
                           refine=True, joint_refit=True)
    residual = cleaned['residual']
    reconstruction_error = float(np.max(np.abs(flux - cleaned['mean'] -
                                                cleaned['model'] - residual)))
    assert reconstruction_error < 1e-7
    history = cleaned['history']
    rss = np.array([h['weighted_rss'] for h in history])
    assert np.all(np.diff(rss) <= max(1.0, float(np.max(rss))) * 1e-10)
    clean_spec = spectrum(time_days, residual, search_grid, time_unit='d')
    overview_clean = spectrum(time_days, residual, overview_grid, time_unit='d')

    _spectrum_csv(results / 'extra1_star03_original_spectrum.csv', raw)
    _spectrum_csv(results / 'extra1_star03_residual_spectrum.csv', clean_spec)
    _spectrum_csv(results / 'extra1_star03_overview_original.csv', overview_raw)
    _spectrum_csv(results / 'extra1_star03_overview_residual.csv', overview_clean)
    _rows_csv(results / 'extra1_star03_components.csv', cleaned['components'])
    _rows_csv(results / 'extra1_star03_history.csv', history)
    np.savetxt(results / 'extra1_star03_time_series.csv',
               np.column_stack([time_days, flux, cleaned['model'], residual]),
               delimiter=',', header='time_day,original_ppm,harmonic_model_ppm,residual_ppm', comments='')

    original_rms = float(np.std(flux))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    peak_raw, peak_clean = int(np.argmax(raw['power'])), int(np.argmax(clean_spec['power']))
    summary = {
        'extra_idea': 'Lecture 1 idea 1: CLEAN the Kepler delta Scuti star03 spectrum',
        'method': 'Iterative sine-wave subtraction, local frequency refinement, and joint harmonic coefficient/intercept refitting; actual timestamps, equal weights.',
        'source_filename': entry['filename'], 'source_sha256': entry['sha256'],
        'observations': int(len(flux)), 'span_days': float(np.ptp(time_days)),
        'resolution_1_over_T_microHz': float(resolution),
        'median_cadence_seconds': float(np.median(np.diff(time_days)) * 86400),
        'overview_grid_microHz': {'min': 1.0, 'max': 4000.0, 'step': 0.5},
        'clean_search_grid_microHz': {'min': float(search_grid[0]), 'max': float(search_grid[-1]),
                                     'step': float(step), 'count': int(len(search_grid))},
        'background_band_microHz': [2000.0, 4000.0],
        'original_background_mean_amplitude_ppm': background,
        'amplitude_threshold_ppm': threshold, 'threshold_multiplier': 4.0,
        'max_components': 20, 'components_removed': len(cleaned['components']),
        'stop_reason': cleaned['stop_reason'], 'fitted_mean_ppm': float(cleaned['mean']),
        'original_rms_ppm': original_rms, 'residual_rms_ppm': residual_rms,
        'fraction_time_variance_removed': float(1 - (residual_rms / original_rms)**2),
        'original_search_peak_microHz': float(search_grid[peak_raw]),
        'original_search_peak_amplitude_ppm': float(raw['amplitude'][peak_raw]),
        'residual_search_peak_microHz': float(search_grid[peak_clean]),
        'residual_search_peak_amplitude_ppm': float(clean_spec['amplitude'][peak_clean]),
        'reconstruction_max_abs_error_ppm': reconstruction_error,
        'rms_monotonic': True,
        'leading_components': sorted(cleaned['components'], key=lambda c: c['amplitude'], reverse=True)[:5],
        'limitations': 'The 50-600 microHz search targets the dominant pulsations and harmonic-region peaks, not every variability component. The wide scan is a diagnostic, not a hard irregular-sampling Nyquist limit. The fixed 4-times-background stopping threshold is heuristic, not a false-alarm probability; the iteration cap bounds model complexity. Removed sinusoids are empirical terms, not automatically independent stellar modes. Slow variation, out-of-band signals, amplitude/phase variability and noise may remain.',
        'elapsed_seconds': time.perf_counter() - started,
    }
    (results / 'extra1_star03_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig = plt.figure(figsize=(11.3, 6.3), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    top = fig.add_subplot(gs[0, :])
    top.semilogy(search_grid, raw['amplitude'], color='#0072B2', lw=0.75, label='Original')
    top.semilogy(search_grid, clean_spec['amplitude'], color='#D55E00', lw=0.75, label='CLEAN residual')
    top.axhline(threshold, color='0.5', ls=':', lw=0.9, label='Fixed amplitude threshold')
    top.set(xlabel='Frequency (microHz)', ylabel='Fitted amplitude (ppm)',
            title=f'(a) star03: {len(cleaned["components"])} fitted components removed', xlim=(50, 600))
    top.legend(fontsize=8, ncol=3, loc='upper right')
    lower = fig.add_subplot(gs[1, 0])
    selected = relative_days <= 0.5
    lower.plot(relative_days[selected], flux[selected] - cleaned['mean'], color='#0072B2',
               lw=0.9, label='Original minus fitted mean')
    lower.plot(relative_days[selected], cleaned['model'][selected], color='#009E73',
               ls='--', lw=0.85, label='Fitted oscillations')
    lower.plot(relative_days[selected], residual[selected], color='#D55E00', lw=0.8, label='Residual')
    lower.set(xlabel='Time from first observation (day)', ylabel='Photometry (ppm)',
              title='(b) First 0.5 day; all data fitted')
    lower.legend(fontsize=7.5, loc='upper right')
    conv = fig.add_subplot(gs[1, 1])
    iterations = [0] + [h['iteration'] for h in history]
    rmss = [original_rms] + [h['weighted_rms'] for h in history]
    conv.semilogy(iterations, rmss, 'o-', color='#0072B2', lw=1.1, ms=3)
    conv.set(xlabel='Components removed', ylabel='Residual RMS (ppm)',
             title='(c) Iterative reduction; fixed stopping rule')
    for ax in [top, lower, conv]:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(figures / '07_star03_clean.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return summary


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
