"""Extra idea 2: Gaussian high-pass filtering of Sun/GOLF observations.

Run ``python extra_highpass.py`` from the project folder, or call
``run_analysis(root)`` from the notebook. Inputs are bundled; no download is
performed. Existing required-task spectra are not overwritten.
"""

from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from filtering import gaussian_highpass, highpass_amplitude_response
from time_series import spectrum


def run_analysis(root=None, show=False):
    """Write the optional experiment outputs and return its summary.

    ``show=False`` closes the figure after saving, so notebooks can display
    the saved PNG exactly once. Set ``show=True`` for an interactive session.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results, figures = root / 'results', root / 'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((root / 'data/source_manifest.json').read_text())
    entry = next(item for item in manifest['datasets'] if item['dataset_label'] == 'Sun (GOLF)')
    path = root / 'data/raw' / entry['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    sun = np.loadtxt(path)
    cutoffs = [20.0, 100.0, 1000.0]
    filtered = [gaussian_highpass(sun[:, 0], sun[:, 1], cutoff, time_unit='d')
                for cutoff in cutoffs]
    use = np.logical_and.reduce([r['interior'] for r in filtered])
    t = sun[use, 0]
    signals = [sun[use, 1]] + [r['highpass'][use] for r in filtered]
    labels = ['Unfiltered'] + [f'{c / 1000:g} mHz high-pass' for c in cutoffs]
    keys = ['unfiltered', '20uhz', '100uhz', '1000uhz']
    np.savetxt(results / 'highpass_time_series.csv',
               np.column_stack([t] + signals), delimiter=',',
               header='time_day,unfiltered_m_per_s,hp20_m_per_s,hp100_m_per_s,hp1000_m_per_s', comments='')
    # Common grid and identical timestamps keep the spectral comparison fair.
    frequencies = np.arange(2.0, 4500.01, 1.0)
    spectra = []
    for key, y in zip(keys, signals):
        result = spectrum(t, y, frequencies, time_unit='d')
        spectra.append(result)
        columns = list(result)
        np.savetxt(results / f'highpass_spectrum_{key}.csv',
                   np.column_stack([result[c] for c in columns]), delimiter=',',
                   header=','.join(columns), comments='')
    bands = {'low_2_20_uhz': (2.0, 20.0), 'low_20_100_uhz': (20.0, 100.0),
             'oscillation_2000_4500_uhz': (2000.0, 4500.0)}
    rows = []
    for label, cutoff, r in zip(labels, [None] + cutoffs, spectra):
        row = {'scheme': label, 'cutoff_microHz': cutoff}
        for name, (left, right) in bands.items():
            band = (frequencies >= left) & (frequencies <= right)
            row[name + '_power_ratio'] = float(np.sum(r['power'][band]) /
                                               np.sum(spectra[0]['power'][band]))
        rows.append(row)
    with (results / 'highpass_comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Gains measured on the real timestamps, independently using lstsq.
    relative = (sun[:, 0] - sun[0, 0]) * 86400
    gain_rows = []
    for cutoff in cutoffs:
        phase = 2 * np.pi * cutoff * 1e-6 * relative
        trial = gaussian_highpass(sun[:, 0], np.sin(phase + 0.4), cutoff, time_unit='d')
        design = np.column_stack((np.sin(phase[use]), np.cos(phase[use]), np.ones(use.sum())))
        coefficients = np.linalg.lstsq(design, trial['highpass'][use], rcond=None)[0]
        gain_rows.append({'cutoff_microHz': cutoff, 'measured_cutoff_amplitude_gain':
                          float(np.hypot(*coefficients[:2])),
                          'expected_cutoff_amplitude_gain': float(1 / np.sqrt(2))})
    assert max(abs(r['measured_cutoff_amplitude_gain'] - 1 / np.sqrt(2))
               for r in gain_rows) < 0.003

    colors = ['#333333', '#0072B2', '#D55E00', '#009E73']
    fig = plt.figure(figsize=(11.3, 6.2), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    response_grid = np.geomspace(2, 4500, 500)
    for cutoff, label, color in zip(cutoffs, labels[1:], colors[1:]):
        axes[0].semilogx(response_grid, highpass_amplitude_response(response_grid, cutoff),
                        label=label, color=color)
        axes[0].plot(cutoff, 1 / np.sqrt(2), 'o', color=color, ms=4)
    axes[0].axhline(1 / np.sqrt(2), color='0.65', lw=0.8, ls=':')
    axes[0].set(xlabel='Frequency (microHz)', ylabel='Amplitude gain',
                title='(a) Defined continuous-kernel response', xlim=(2, 4500), ylim=(0, 1.04))
    axes[0].legend(fontsize=8, loc='upper left')
    for label, r, color in zip(labels, spectra, colors):
        axes[1].loglog(frequencies, r['power'], label=label, color=color, lw=0.65, alpha=0.85)
        axes[2].plot(frequencies, r['power'], label=label, color=color, lw=0.75, alpha=0.75)
    axes[1].set(xlabel='Frequency (microHz)', ylabel='Fitted power (m2/s2)',
                title='(b) Low-frequency suppression', xlim=(2, 1500))
    axes[2].set(xlabel='Frequency (microHz)', ylabel='Fitted power (m2/s2)',
                title='(c) Solar oscillation band: identical retained timestamps', xlim=(2000, 4500))
    axes[2].legend(fontsize=8, ncol=4, loc='upper right')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(figures / '06_solar_highpass.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    summary = {
        'extra_idea': 'Lecture 1 p.65 idea 2: Sun/GOLF high-pass comparison',
        'method': 'Original timestamps; normalized Gaussian smoothing with trapezoidal time-integration weights, then subtraction.',
        'cutoff_definition': 'High-pass amplitude H(fc)=1/sqrt(2), i.e. -3.0103 dB; H(f)=1-exp(-2*pi^2*sigma^2*f^2).',
        'cutoffs_microHz': cutoffs,
        'sigma_seconds': [r['sigma_s'] for r in filtered],
        'kernel_truncate_sigma': 5.0,
        'common_edge_exclusion_seconds': max(r['half_support_s'] for r in filtered),
        'original_samples': len(sun), 'retained_samples': int(use.sum()),
        'removed_samples': int((~use).sum()),
        'retained_span_days': float(np.ptp(t)),
        'frequency_grid_microHz': {'min': 2.0, 'max': 4500.0, 'step': 1.0, 'count': len(frequencies)},
        'gain_checks': gain_rows, 'comparison': rows,
        'limitations': 'The response is a continuous, infinite-record approximation. The finite-kernel, nearly regular data were checked at the actual timestamps. Common boundary trimming changes the observing window. Band power ratios are descriptive sums of sampled amplitude-squared spectra, not PSD integrals, physical variances, or formal detection significances.',
        'elapsed_seconds': time.perf_counter() - started,
    }
    (results / 'highpass_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


# Backward-compatible descriptive name for notebook callers.
run_highpass = run_analysis


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
