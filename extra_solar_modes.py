"""Extra idea 3: compare three narrow-band SoHO/GOLF time series.

Run ``python extra_solar_modes.py``. Data, code and outputs are bundled;
the experiment does not need a network connection or Extra idea 4 results.
"""

from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from bandpass import harmonic_bandpass
from time_series import spectrum


def export_columns(path, columns):
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=',',
               header=','.join(columns), comments='')


def export_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def evaluate_model(seconds, projection):
    phase = 2*np.pi*1e-6*np.asarray(seconds)[:, None]*projection['frequency_uhz'][None, :]
    return np.sin(phase)@projection['alpha'] + np.cos(phase)@projection['beta']


def run_analysis(root=None, show=False):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results, figures = root/'results', root/'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    start = time.perf_counter()
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    entry = next(e for e in manifest['datasets'] if e['dataset_label'] == 'Sun (GOLF)')
    raw = root/'data/raw'/entry['filename']
    if hashlib.sha256(raw.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Input data do not match the recorded SHA-256')
    times, velocity = np.loadtxt(raw).T
    seconds = (times-times[0])*86400.0
    days = seconds/86400.0
    interior = (days >= 1.0) & (days <= days[-1]-1.0)
    zoom = (days >= 10.0) & (days <= 10.0+1/24)
    display_seconds = 10*86400+np.linspace(0, 3600, 1201)
    display_minutes = (display_seconds-10*86400)/60
    day_centres = np.arange(0.5, np.floor(days[-1])+1, 1.0)
    rows, gain_rows, models, curves, daily_values = [], [], [], [], []

    for nominal in [2000.0, 3000.0, 4000.0]:
        tag = str(int(nominal))
        # Direct local searches avoid dependence on optional earlier outputs.
        coarse_grid = nominal-50+np.arange(501)*0.2
        coarse = spectrum(times, velocity, coarse_grid, time_unit='d')
        # Exclude rising boundary shoulders whose peaks lie outside the search.
        local_indices = np.flatnonzero((coarse['power'][1:-1] > coarse['power'][:-2]) &
                                       (coarse['power'][1:-1] >= coarse['power'][2:]))+1
        if not len(local_indices):
            raise ValueError(f'No interior local peak found near {nominal} microHz')
        peak_index = local_indices[np.argmax(coarse['power'][local_indices])]
        candidate = float(coarse_grid[peak_index])
        fine_grid = candidate+np.arange(-25, 26)*0.01
        fine_grid = fine_grid[(fine_grid >= nominal-50) & (fine_grid <= nominal+50)]
        fine = spectrum(times, velocity, fine_grid, time_unit='d')
        centre = float(fine_grid[np.argmax(fine['power'])])
        band = [centre-5.0, centre+5.0]
        projection = harmonic_bandpass(times, velocity, band, time_unit='d')
        model = projection['model']
        np.testing.assert_allclose(velocity,
                                   projection['mean']+model+projection['residual'],
                                   rtol=0, atol=1e-12)
        models.append(model)
        curves.append(evaluate_model(display_seconds, projection))
        daily = np.array([rms(model[(days >= d-.5) & (days < d+.5)])
                          for d in day_centres])
        daily_values.append(daily)
        export_columns(results/f'solar_modes_coarse_search_{tag}.csv', coarse)
        export_columns(results/f'solar_modes_fine_search_{tag}.csv', fine)
        export_columns(results/f'solar_modes_coefficients_{tag}.csv',
                       {k: projection[k] for k in ['frequency_uhz', 'alpha', 'beta', 'amplitude']})
        rows.append({
            'nominal_microHz': nominal,
            'selected_peak_microHz': centre,
            'period_seconds': 1e6/centre,
            'period_minutes': 1e6/centre/60,
            'lower_microHz': band[0], 'upper_microHz': band[1],
            'frequency_spacing_microHz': projection['grid_spacing_uhz'],
            'harmonic_components': len(projection['frequency_uhz']),
            'full_record_rms_m_per_s': rms(model),
            'interior_rms_m_per_s': rms(model[interior]),
            'interior_daily_rms_min_m_per_s': float(daily[1:-1].min()),
            'interior_daily_rms_max_m_per_s': float(daily[1:-1].max()),
            'design_condition_number': projection['condition_number'],
        })

        # Measure leakage and finite-window behavior at the real timestamps.
        for position, frequency in [('centre', centre), ('upper_edge', band[1]),
                                    ('outside_band', centre+30.0)]:
            phase = 2*np.pi*frequency*1e-6*seconds
            injected = np.sin(phase+0.37)
            retained = harmonic_bandpass(times, injected, band, time_unit='d')['model']
            design = np.column_stack((np.sin(phase[interior]),
                                      np.cos(phase[interior]), np.ones(interior.sum())))
            coefficients = np.linalg.lstsq(design, retained[interior], rcond=None)[0]
            gain_rows.append({
                'nominal_microHz': nominal, 'position': position,
                'injected_microHz': frequency,
                'full_record_rms_gain': rms(retained)/rms(injected),
                'interior_fitted_amplitude_gain': float(np.hypot(coefficients[0], coefficients[1])),
            })

    assert all(abs(r['interior_fitted_amplitude_gain']-1) < 0.01
               for r in gain_rows if r['position'] == 'centre')
    assert all(r['full_record_rms_gain'] < 0.04
               for r in gain_rows if r['position'] == 'outside_band')
    tags = ['2000', '3000', '4000']
    export_columns(results/'solar_modes_time_series.csv',
                   {'time_day': times, **{f'band_{tag}_m_per_s': m for tag, m in zip(tags, models)}})
    export_columns(results/'solar_modes_daily_rms.csv',
                   {'day_from_start': day_centres,
                    **{f'band_{tag}_rms_m_per_s': d for tag, d in zip(tags, daily_values)}})
    export_rows(results/'solar_modes_comparison.csv', rows)
    export_rows(results/'solar_modes_injection_checks.csv', gain_rows)

    fig, axes = plt.subplots(3, 2, figsize=(11.3, 8), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73']
    waveform_limit = 1.12*max(float(np.max(np.abs(c))) for c in curves)
    daily_limit = 1.12*max(float(np.max(d)) for d in daily_values)
    for k, (row, color) in enumerate(zip(rows, colors)):
        axes[k, 0].plot(display_minutes, curves[k], color=color, lw=1.1)
        axes[k, 0].plot((days[zoom]-10)*1440, models[k][zoom], '.', color=color, ms=2.5)
        axes[k, 0].axhline(0, color='0.8', lw=0.6, zorder=0)
        axes[k, 0].set(xlim=(0, 60), ylim=(-waveform_limit, waveform_limit),
                       xlabel='Minutes from day 10', ylabel='Velocity (m/s)',
                       title=f'{row["selected_peak_microHz"]/1000:.4f} mHz | period {row["period_minutes"]:.2f} min')
        axes[k, 1].plot(day_centres, daily_values[k], 'o-', color=color, ms=3, lw=1)
        axes[k, 1].axvspan(0, 1, color='0.92', zorder=0)
        axes[k, 1].axvspan(days[-1]-1, days[-1], color='0.92', zorder=0)
        axes[k, 1].set(xlim=(0, 30), ylim=(0, daily_limit),
                       xlabel='Time from first observation (day)', ylabel='Daily RMS (m/s)',
                       title=f'Interior RMS = {row["interior_rms_m_per_s"]:.4f} m/s')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Three solar bands, each 10 microHz wide | common vertical scales within columns', fontsize=12)
    fig.savefig(figures/'08_solar_modes_comparison.png', dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    summary = {
        'extra_idea': 3, 'source_filename': entry['filename'], 'source_sha256': entry['sha256'],
        'sample_count': len(times), 'span_days': float(days[-1]),
        'search_half_width_microHz': 50.0, 'coarse_search_step_microHz': 0.2,
        'peak_selection': 'Strongest interior local maximum on the coarse grid, refined within +/-0.25 microHz while constrained to the search interval; boundary shoulders excluded.',
        'refined_search_step_microHz': 0.01, 'band_half_width_microHz': 5.0,
        'interior_day_range': [1.0, float(days[-1]-1)],
        'interior_sample_count': int(interior.sum()),
        'method': 'Joint constant + sine/cosine least-squares projection on positive k/T frequencies inside the band; T=N times mean observed spacing. Return only the harmonic model.',
        'relation_to_lecture': 'Harmonic reconstruction using a joint finite-band fit; not the oversampled independent-coefficient/window-area normalization formula.',
        'bands': rows, 'injection_checks': gain_rows,
        'rms_ratio_3_to_2': rows[1]['interior_rms_m_per_s']/rows[0]['interior_rms_m_per_s'],
        'rms_ratio_3_to_4': rows[1]['interior_rms_m_per_s']/rows[2]['interior_rms_m_per_s'],
        'figure_note': 'Same physical vertical scale within each column. Curves densely evaluate fitted models for display; dots and all statistics use original timestamps. Shaded endpoint days are omitted from the interior statistics.',
        'limitations': 'Finite-record band projection has leakage and non-sharp edges. A 10 microHz band can contain multiple nearby components; selecting a spectral peak does not establish a unique physical eigenmode. Variations in daily RMS can arise from beating, noise or finite coherence. Endpoint exclusion is descriptive, not proof of zero edge bias. No individual mode lifetime is inferred.',
        'elapsed_seconds': time.perf_counter()-start,
    }
    (results/'solar_modes_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run_analysis(), indent=2))
