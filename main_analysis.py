"""Reproduce the main note: harmonic spectra, stellar data and spectral windows.

Run ``python main_analysis.py`` after downloading the source observations.
Only NumPy and Matplotlib are required. The numerical routines are in
``time_series.py``; this module defines the experiments and their figures.
"""

from pathlib import Path
import csv
import hashlib
import json
import platform
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from time_series import spectrum, spectral_window


DAY = 86400.0
MAIN_DATASETS = ('Sun (GOLF)', 'Procyon (multisite)')


def export_spectrum(directory, name, result):
    """Write the columns returned by the numerical spectrum implementation."""
    columns = list(result)
    np.savetxt(directory/name, np.column_stack([result[k] for k in columns]),
               delimiter=',', header=','.join(columns), comments='')


def export_table(directory, name, rows):
    with (directory/name).open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name, show):
    fig.savefig(directory/name, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def fwhm(x, y):
    """Width of the connected half-maximum lobe around the highest point."""
    peak = int(np.nanargmax(y))
    level = y[peak]/2
    left = np.flatnonzero(y[:peak] <= level)
    right = np.flatnonzero(y[peak+1:] <= level)
    if not len(left) or not len(right):
        return np.nan
    i, j = left[-1], peak+1+right[0]
    xl = x[i]+(level-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    xr = x[j-1]+(level-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1])
    return float(xr-xl)


def load_observations(root):
    """Load and verify only the two datasets needed by the main analysis."""
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    data, records = {}, []
    for label in MAIN_DATASETS:
        entry = next(item for item in manifest['datasets'] if item['dataset_label'] == label)
        path = root/'data/raw'/entry['filename']
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path.name}; run python download_data.py first.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch for {path.name}')
        values = np.loadtxt(path)
        data[label] = values
        records.append({'dataset': label, 'points': len(values),
                        'span_days': float(np.ptp(values[:, 0])),
                        'columns': int(values.shape[1])})
    return data, records


def verify_harmonics(root, results, figures, show):
    """Vary frequency, amplitude and duration independently."""
    frequency_step = 0.25
    frequencies = np.arange(400.0, 2800.0+0.5*frequency_step, frequency_step)
    t = np.arange(0, 2*DAY, 60.0)
    y = 1.5*np.sin(2*np.pi*1600e-6*t+0.7)
    example = spectrum(t, y, frequencies, time_unit='s', center=False)
    export_spectrum(results, 'example_spectrum.csv', example)
    np.savetxt(root/'data/synthetic.csv', np.column_stack([t, y]), delimiter=',',
               header='time_s,value', comments='')
    verification = []
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4), layout='constrained')
    for injected in [800.0, 1600.0, 2400.0]:
        signal = 1.5*np.sin(2*np.pi*injected*1e-6*t+0.7)
        result = spectrum(t, signal, frequencies, center=False)
        export_spectrum(results, f'synthetic_frequency_{injected:.0f}.csv', result)
        peak = np.argmax(result['power'])
        verification.append({'test': 'frequency', 'input': injected,
                             'measured': float(result['frequency_uhz'][peak]),
                             'expected': injected, 'unit': 'microHz'})
        axes[0].plot(frequencies, result['power'], lw=1.0, label=f'{injected:.0f} microHz')
    axes[0].set(xlabel='Frequency (microHz)', ylabel='Power (arbitrary units squared)',
                title='Frequency: peak position')
    axes[0].legend(fontsize=8)

    local = np.arange(1570, 1630.01, 0.1)
    for amplitude in [0.5, 1.0, 2.0]:
        result = spectrum(t, amplitude*np.sin(2*np.pi*1600e-6*t+0.7), local, center=False)
        export_spectrum(results, f'synthetic_amplitude_{amplitude:.1f}.csv', result)
        verification.append({'test': 'amplitude squared', 'input': amplitude,
                             'measured': float(np.max(result['power'])),
                             'expected': amplitude*amplitude, 'unit': 'power'})
        axes[1].plot(local-1600, result['power'], label=f'a = {amplitude:g}')
    axes[1].set(xlabel='Frequency offset (microHz)', ylabel='Power (arbitrary units squared)',
                title='Amplitude: peak power')
    axes[1].legend(fontsize=8)

    for days in [1, 2, 4]:
        duration = days*DAY
        td = np.arange(0, duration, 60.0)
        offset = np.linspace(-6e6/duration, 6e6/duration, 1201)
        result = spectrum(td, np.sin(2*np.pi*2000e-6*td+0.7), 2000+offset, center=False)
        export_spectrum(results, f'synthetic_duration_{days}d.csv', result)
        verification.append({'test': 'duration / FWHM', 'input': days,
                             'measured': fwhm(offset, result['power']),
                             'expected': 0.8859e6/duration, 'unit': 'microHz'})
        axes[2].plot(offset, result['power'], label=f'{days} day(s)')
    axes[2].set(xlabel='Frequency offset (microHz)', ylabel='Power (arbitrary units squared)',
                title='Duration: peak width', xlim=(-20, 20))
    axes[2].legend(fontsize=8)
    save_figure(fig, figures, '01_harmonic_verification.png', show)
    export_table(results, 'harmonic_verification.csv', verification)
    width_error = max(abs(row['measured']/row['expected']-1)
                      for row in verification if row['test'] == 'duration / FWHM')
    assert width_error < 0.01
    assert max(abs(row['measured']-row['expected']) for row in verification[:3]) <= frequency_step/2
    assert max(abs(row['measured']-row['expected']) for row in verification[3:6]) < 1e-8
    return verification, width_error


def analyse_sun(sun, results, figures, show):
    """Compare observed solar peak groups with coherent reference windows."""
    grid = np.arange(1500.0, 4500.01, 0.2)
    solar = spectrum(sun[:, 0], sun[:, 1], grid, time_unit='d')
    export_spectrum(results, 'sun_overview.csv', solar)
    fig, ax = plt.subplots(figsize=(10.4, 2.8), layout='constrained')
    ax.plot(grid, solar['amplitude'], lw=0.7)
    ax.set(xlabel='Frequency (microHz)', ylabel='Amplitude (m/s)',
           title=f'Sun / GOLF: all {len(sun):,} observations')
    save_figure(fig, figures, '02_sun_overview.png', show)

    shapes = []
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4), layout='constrained')
    for ax, nominal in zip(axes, [2964.0, 3098.0, 3233.0]):
        trial = nominal+np.linspace(-4, 4, 401)
        initial = spectrum(sun[:, 0], sun[:, 1], trial, time_unit='d')
        peak_frequency = trial[np.argmax(initial['power'])]
        offsets = np.linspace(-3, 3, 601)
        local_grid = peak_frequency+offsets
        observed = spectrum(sun[:, 0], sun[:, 1], local_grid, time_unit='d')
        window = spectral_window(sun[:, 0], local_grid, peak_frequency, time_unit='d')
        export_spectrum(results, f'sun_group_{nominal:.0f}.csv', observed)
        export_spectrum(results, f'sun_coherent_window_{nominal:.0f}.csv', window)
        outside = np.abs(offsets) > 0.6000001
        observed_wings = np.sum(observed['power'][outside])/np.sum(observed['power'])
        coherent_wings = np.sum(window['power'][outside])/np.sum(window['power'])
        shapes.append({'peak_microHz': float(peak_frequency),
                       'peak_amplitude_m_per_s': float(np.max(observed['amplitude'])),
                       'observed_wing_fraction': float(observed_wings),
                       'coherent_wing_fraction': float(coherent_wings)})
        ax.plot(offsets, observed['power']/np.max(observed['power']), label='Observed', lw=1.2)
        ax.plot(offsets, window['power']/np.max(window['power']), '--', label='Coherent window', lw=1.3)
        ax.axvline(-0.6, color='0.7', lw=0.7)
        ax.axvline(0.6, color='0.7', lw=0.7)
        ax.set(xlabel='Frequency offset (microHz)', ylabel='Peak-normalized power',
               title=f'Group near {peak_frequency:.2f} microHz', xlim=(-3, 3), ylim=(0, 1.08))
    axes[0].legend(fontsize=8)
    save_figure(fig, figures, '03_sun_peak_shapes.png', show)
    export_table(results, 'solar_peak_shapes.csv', shapes)
    return shapes


def analyse_procyon(procyon, results, figures, show):
    """Measure the background and window change caused by supplied SNR weights."""
    grid = np.arange(200.0, 3000.01, 0.25)
    plain = spectrum(procyon[:, 0], procyon[:, 1], grid, time_unit='d')
    weighted = spectrum(procyon[:, 0], procyon[:, 1], grid, weights=procyon[:, 2], time_unit='d')
    export_spectrum(results, 'procyon_unweighted.csv', plain)
    export_spectrum(results, 'procyon_snr_weighted.csv', weighted)
    offsets = np.linspace(-30, 30, 1201)
    win_plain = spectral_window(procyon[:, 0], 1000+offsets, 1000, time_unit='d')
    win_weighted = spectral_window(procyon[:, 0], 1000+offsets, 1000,
                                   weights=procyon[:, 2], time_unit='d')
    export_spectrum(results, 'procyon_window_unweighted.csv', win_plain)
    export_spectrum(results, 'procyon_window_snr_weighted.csv', win_weighted)
    noise_band = (grid >= 2200) & (grid <= 3000)
    comparisons = []
    for label, result, window in [('Equal weights', plain, win_plain),
                                  ('SNR weights', weighted, win_weighted)]:
        alias = spectral_window(procyon[:, 0], [1000+1e6/DAY], 1000,
                                weights=None if label == 'Equal weights' else procyon[:, 2],
                                time_unit='d')
        comparisons.append({'scheme': label,
                            'mean_noise_amplitude_m_per_s': float(np.mean(result['amplitude'][noise_band])),
                            'mean_noise_power_m2_per_s2': float(np.mean(result['power'][noise_band])),
                            'window_FWHM_microHz': fwhm(offsets, window['power']),
                            'daily_alias_relative_power': float(alias['power'][0])})

    fig = plt.figure(figsize=(11.0, 6.3), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for label, result, color in [('Equal weights', plain, '#0072B2'),
                                  ('SNR weights', weighted, '#D55E00')]:
        axes[0].plot(grid, result['amplitude'], label=label, color=color, lw=0.65, alpha=0.85)
        axes[1].plot(grid[noise_band], result['amplitude'][noise_band], color=color, lw=0.65, alpha=0.8)
    axes[0].set(xlim=(200, 2000), xlabel='Frequency (microHz)', ylabel='Amplitude (m/s)',
                title='Oscillation region')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Frequency (microHz)', ylabel='Amplitude (m/s)',
                title='Fixed high-frequency comparison band')
    axes[2].plot(offsets, win_plain['power'], label='Equal weights')
    axes[2].plot(offsets, win_weighted['power'], label='SNR weights')
    axes[2].set(xlabel='Frequency offset from 1000 microHz', ylabel='Relative window power',
                title='The same observations, with different effective windows', xlim=(-18, 18))
    axes[2].legend(fontsize=8)
    save_figure(fig, figures, '04_procyon_weights.png', show)
    export_table(results, 'procyon_weight_comparison.csv', comparisons)
    reduction = 1-comparisons[1]['mean_noise_amplitude_m_per_s']/comparisons[0]['mean_noise_amplitude_m_per_s']
    return comparisons, reduction


def analyse_gaps(results, figures, show):
    """Test first minima and sidelobes for one or two uniformly sampled segments."""
    duration, cadence = DAY, 120.0
    segment = np.arange(0, duration, cadence)
    reference = 0.5*(1e6/(2*cadence))
    offsets = np.linspace(-4e6/duration, 4e6/duration, 4001)
    grid = reference+offsets
    cases = [('Single segment', None)]+[(f'Gap = {gap:g} D', gap) for gap in [0, 0.5, 1, 2, 4]]
    comparisons = []
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.3), sharex=True, sharey=True, layout='constrained')
    for ax, (label, gap) in zip(axes.flat, cases):
        if gap is None:
            times = segment
            predicted_minimum = 1e6/duration
            predicted = np.sinc(duration*offsets*1e-6)**2
        else:
            times = np.concatenate([segment, segment+duration*(1+gap)])
            predicted_minimum = 1e6/(2*duration*(1+gap))
            predicted = np.sinc(duration*offsets*1e-6)**2*np.cos(np.pi*duration*(1+gap)*offsets*1e-6)**2
        window = spectral_window(times, grid, reference)
        name = 'single' if gap is None else f'gap_{gap:g}'
        export_spectrum(results, f'window_{name}.csv', window)
        minima = np.flatnonzero((window['power'][1:-1] <= window['power'][:-2]) &
                               (window['power'][1:-1] < window['power'][2:]) &
                               (offsets[1:-1] > 0))+1
        index = int(minima[0])
        left, centre, right = window['power'][index-1:index+2]
        displacement = 0.5*(left-right)/(left-2*centre+right)
        measured_minimum = offsets[index]+displacement*(offsets[1]-offsets[0])
        outer = (np.abs(offsets) > measured_minimum) & (np.abs(offsets) <= 35)
        comparisons.append({'case': label, 'points': len(times),
                            'first_minimum_microHz': float(measured_minimum),
                            'predicted_minimum_microHz': predicted_minimum,
                            'largest_outer_peak_power': float(np.max(window['power'][outer])),
                            'window_FWHM_microHz': fwhm(offsets, window['power']),
                            'outer_search_half_width_microHz': 35.0})
        ax.plot(offsets, window['power'], label='Numerical', lw=1.3)
        ax.plot(offsets, predicted, '--', label='Continuous approximation', lw=0.9)
        ax.set(title=label, xlim=(-35, 35), ylim=(-0.03, 1.08),
               xlabel='Frequency offset (microHz)', ylabel='Relative window power')
    axes[0, 0].legend(fontsize=7.5)
    save_figure(fig, figures, '05_gap_windows.png', show)
    export_table(results, 'gap_window_comparison.csv', comparisons)
    error = max(abs(row['first_minimum_microHz']-row['predicted_minimum_microHz']) for row in comparisons)
    assert error <= 1.1*(offsets[1]-offsets[0])
    return comparisons, error


def run_analysis(root=None, show=False):
    """Recompute all required experiments and write a main-only summary."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results, figures = root/'results', root/'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    started = time.perf_counter()
    datasets, data_summary = load_observations(root)
    export_table(results, 'data_summary.csv', data_summary)
    sun, procyon = (datasets[label] for label in MAIN_DATASETS)
    style = {'figure.dpi': 110, 'savefig.dpi': 160, 'font.size': 10,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.labelsize': 10, 'axes.titlesize': 11,
             'axes.prop_cycle': matplotlib.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7'])}
    with plt.rc_context(style):
        verification, width_error = verify_harmonics(root, results, figures, show)
        solar_shapes = analyse_sun(sun, results, figures, show)
        weight_comparison, noise_reduction = analyse_procyon(procyon, results, figures, show)
        gap_results, gap_error = analyse_gaps(results, figures, show)
    summary = {
        'analysis': 'Main note: harmonic spectra, solar peak shapes, Procyon weights and observing gaps',
        'python': platform.python_version(), 'numpy': np.__version__,
        'matplotlib': matplotlib.__version__, 'elapsed_seconds': time.perf_counter()-started,
        'data_summary': data_summary, 'harmonic_verification': verification,
        'solar_resolution_microHz': float(1e6/(np.ptp(sun[:, 0])*DAY)),
        'procyon_mean_noise_amplitude_reduction': noise_reduction,
        'max_synthetic_width_relative_error': width_error,
        'max_gap_first_minimum_absolute_error_microHz': gap_error,
        'solar_shapes': solar_shapes, 'procyon_weight_comparison': weight_comparison,
        'gap_results': gap_results,
    }
    (results/'main_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
