"""Recalculate the required analysis and all five extra ideas in order."""
from importlib import import_module
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    for name in ('main_analysis', 'extra_star03', 'extra_highpass',
                 'extra_solar_modes', 'extra_clean_modes', 'extra_lmi'):
        print(f'Running {name} ...', flush=True)
        import_module(name).run_analysis(root, show=False)
    print('Completed. Results and figures have been updated.')


if __name__ == '__main__':
    main()
