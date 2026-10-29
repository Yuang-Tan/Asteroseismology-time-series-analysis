"""Extra idea 4: CLEAN band-pass reconstruction using the exact Extra 3 bands.

Run ``python extra_clean_modes.py`` from the supplied standalone bundle.
The raw data and the frozen Extra 3 reference files are included. Each band
is analysed independently; the returned band-pass series is the sum of the
retained CLEAN components, not the subtraction residual.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from bandpass import harmonic_bandpass
from cleaning import clean_series, reconstruct_components
from time_series import spectrum
from extra_solar_modes import export_columns, export_rows, rms, evaluate_model


def run_analysis(root=None, show=False):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results, figures = root/'results', root/'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    entry = next(e for e in manifest['datasets'] if e['dataset_label'] == 'Sun (GOLF)')
    reference_path = results/'solar_modes_summary.json'
    reference = json.loads(reference_path.read_text())
    raw_path = root/'data/raw'/entry['filename']
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_hash != entry['sha256'] or raw_hash != reference['source_sha256']:
        raise ValueError('Raw data differ from the Extra 3 reference')
    times, velocity = np.loadtxt(raw_path).T
    previous = np.genfromtxt(results/'solar_modes_time_series.csv', names=True, delimiter=',')
    np.testing.assert_array_equal(previous['time_day'], times)
    seconds = (times-times[0])*86400.0
    days = seconds/86400.0
    interior = (days >= reference['interior_day_range'][0]) & (days <= reference['interior_day_range'][1])
    assert int(interior.sum()) == reference['interior_sample_count']
    display_seconds = 10*86400+np.linspace(0, 3600, 1201)
    display_minutes = (display_seconds-10*86400)/60
    day_centres = np.arange(0.5, np.floor(days[-1])+1, 1.0)
    time_columns = {'time_day': times, 'original_m_per_s': velocity}
    daily_columns = {'day_from_start': day_centres}
    rows, linear_curves, clean_curves, daily_pairs, spectral_pairs = [], [], [], [], []

    for config in reference['bands']:
        nominal = config['nominal_microHz']
        tag = str(int(nominal))
        centre = config['selected_peak_microHz']
        band = [config['lower_microHz'], config['upper_microHz']]
        np.testing.assert_allclose(band, [centre-5, centre+5], atol=1e-10, rtol=0)
        linear = harmonic_bandpass(times, velocity, band, time_unit='d')
        np.testing.assert_allclose(linear['model'], previous[f'band_{tag}_m_per_s'], atol=1e-12, rtol=0)
        grid = np.linspace(band[0], band[1], 201)
        original_spectrum = spectrum(times, velocity, grid, time_unit='d')
        threshold = 0.10*float(np.max(original_spectrum['amplitude']))
        clean = clean_series(times, velocity, grid, time_unit='d',
                             max_components=12, amplitude_threshold=threshold,
                             refine=True, joint_refit=True,
                             min_separation_uhz=0.25e6/np.ptp(seconds))
        assert all(band[0] <= c['frequency_uhz'] <= band[1] for c in clean['components'])
        accounting_error = float(np.max(np.abs(velocity-clean['mean']-clean['model']-clean['residual'])))
        assert accounting_error < 1e-12
        reconstructed = reconstruct_components(times, clean['components'], time_unit='d', reference_time=times[0])
        np.testing.assert_allclose(reconstructed, clean['model'], rtol=0, atol=1e-12)
        rss = [clean['initial_weighted_rss']]+[h['weighted_rss'] for h in clean['history']]
        assert np.all(np.diff(rss) <= 1e-12*max(rss))
        residual_spectrum = spectrum(times, clean['residual'], grid, time_unit='d')
        retained_spectrum = spectrum(times, clean['model'], grid, time_unit='d')
        for label, values in [('input', original_spectrum), ('residual', residual_spectrum),
                              ('retained', retained_spectrum)]:
            export_columns(results/f'solar_clean_modes_{label}_spectrum_{tag}.csv', values)
        export_rows(results/f'solar_clean_modes_components_{tag}.csv', clean['components'])
        export_rows(results/f'solar_clean_modes_history_{tag}.csv', clean['history'])
        time_columns[f'linear_{tag}_m_per_s'] = linear['model']
        time_columns[f'clean_{tag}_m_per_s'] = clean['model']
        time_columns[f'residual_{tag}_m_per_s'] = clean['residual']
        daily_linear = np.array([rms(linear['model'][(days >= d-.5) & (days < d+.5)]) for d in day_centres])
        daily_clean = np.array([rms(clean['model'][(days >= d-.5) & (days < d+.5)]) for d in day_centres])
        daily_columns[f'linear_{tag}_rms_m_per_s'] = daily_linear
        daily_columns[f'clean_{tag}_rms_m_per_s'] = daily_clean
        daily_pairs.append((daily_linear, daily_clean))
        linear_curves.append(evaluate_model(display_seconds, linear))
        clean_curves.append(reconstruct_components(display_seconds, clean['components'], reference_time=0.0))
        spectral_pairs.append((grid, original_spectrum, residual_spectrum))
        rows.append({
            'nominal_microHz': nominal, 'selected_peak_microHz': centre,
            'lower_microHz': band[0], 'upper_microHz': band[1],
            'components': len(clean['components']), 'max_components': 12,
            'stop_reason': clean['stop_reason'], 'amplitude_threshold_m_per_s': threshold,
            'fitted_mean_m_per_s': clean['mean'],
            'linear_interior_rms_m_per_s': rms(linear['model'][interior]),
            'clean_interior_rms_m_per_s': rms(clean['model'][interior]),
            'interior_correlation': float(np.corrcoef(linear['model'][interior], clean['model'][interior])[0, 1]),
            'interior_difference_rms_m_per_s': rms(clean['model'][interior]-linear['model'][interior]),
            'remaining_band_sampled_power_ratio': float(np.sum(residual_spectrum['power'])/np.sum(original_spectrum['power'])),
            'residual_max_sampled_amplitude_m_per_s': float(np.max(residual_spectrum['amplitude'])),
            'accounting_max_abs_error_m_per_s': accounting_error,
            'rms_monotonic': True,
        })
        print(f'{centre/1000:.4f} mHz: {len(clean["components"])} components; {clean["stop_reason"]}', flush=True)

    export_columns(results/'solar_clean_modes_time_series.csv', time_columns)
    export_columns(results/'solar_clean_modes_daily_rms.csv', daily_columns)
    export_rows(results/'solar_clean_modes_comparison.csv', rows)
    fig, axes = plt.subplots(3, 2, figsize=(11.3, 8), layout='constrained')
    wave_limit = 1.12*max(float(np.max(np.abs(c))) for c in linear_curves+clean_curves)
    daily_limit = 1.12*max(float(np.max(d)) for pair in daily_pairs for d in pair)
    for k, row in enumerate(rows):
        axes[k, 0].plot(display_minutes, linear_curves[k], color='#0072B2', lw=1.4, label='Extra 3: linear band-pass')
        axes[k, 0].plot(display_minutes, clean_curves[k], color='#D55E00', ls='--', lw=1.1, label='Extra 4: CLEAN reconstruction')
        axes[k, 0].set(xlim=(0, 60), ylim=(-wave_limit, wave_limit),
                       xlabel='Minutes from day 10', ylabel='Velocity (m/s)',
                       title=f'{row["selected_peak_microHz"]/1000:.4f} mHz | {row["components"]} CLEAN components')
        for values, color, style in zip(daily_pairs[k], ['#0072B2', '#D55E00'], ['-', '--']):
            axes[k, 1].plot(day_centres, values, color=color, ls=style, marker='o', ms=2.5, lw=1)
        axes[k, 1].axvspan(0, 1, color='0.92', zorder=0)
        axes[k, 1].axvspan(days[-1]-1, days[-1], color='0.92', zorder=0)
        axes[k, 1].set(xlim=(0, 30), ylim=(0, daily_limit),
                       xlabel='Time from first observation (day)', ylabel='Daily RMS (m/s)',
                       title=f'Interior waveform correlation = {row["interior_correlation"]:.3f}')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=10)
    fig.savefig(figures/'09_solar_clean_modes_comparison.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout='constrained')
    for ax, row, (grid, original, residual) in zip(axes, rows, spectral_pairs):
        offset = grid-row['selected_peak_microHz']
        ax.semilogy(offset, original['power'], color='#0072B2', label='Input power')
        ax.semilogy(offset, residual['power'], color='#D55E00', label='Residual power')
        ax.axhline(row['amplitude_threshold_m_per_s']**2, ls=':', color='0.4', label='Amplitude stopping level squared')
        peak_power = float(np.max(original['power']))
        ax.set(xlim=(-5, 5), ylim=(peak_power*1e-5, peak_power*2),
               xlabel='Frequency offset (microHz)', ylabel='Fitted power (m2/s2)',
               title=f'{row["selected_peak_microHz"]/1000:.4f} mHz | residual sampled band power {100*row["remaining_band_sampled_power_ratio"]:.2f}%')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle('Log display floor: 0.00001 times each input peak; full values retained in CSVs', fontsize=10)
    fig.savefig(figures/'09_solar_clean_modes_power.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    summary = {
        'extra_idea': 4, 'source_filename': entry['filename'], 'source_sha256': raw_hash,
        'extra3_reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'sample_count': len(times), 'interior_sample_count': int(interior.sum()),
        'interior_day_range': reference['interior_day_range'],
        'band_half_width_microHz': 5.0, 'search_grid_step_microHz': 0.05,
        'amplitude_stop_fraction': 0.10, 'max_components': 12,
        'method': 'Independent restricted-band time-domain CLEAN per band: strongest residual sinusoid, local frequency refinement and joint sine/cosine/intercept refit; full-amplitude subtraction (gain 1). Retain the harmonic component sum as the band-pass output.',
        'reference_check': 'Linear reconstructions agree with the delivered Extra 3 time series within 1e-12 m/s at identical original timestamps; the exact saved bands are reused.',
        'bands': rows,
        'limitations': 'A data-dependent approximation, not an ideal linear filter or physical mode identification. Component counts depend on thresholds and caps. Two bands hit the cap. Residual band-power sums are descriptive, not PSD variances or significance tests; finite-window cross terms mean model and residual powers are not pointwise additive. No mode lifetimes inferred.',
        'elapsed_seconds': time.perf_counter()-started,
    }
    (results/'solar_clean_modes_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
