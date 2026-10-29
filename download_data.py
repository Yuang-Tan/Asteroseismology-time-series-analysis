"""Download the four original course datasets and verify their recorded hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent


def acquire(root=ROOT, verify_only=False):
    root = Path(root)
    manifest = json.loads((root/'data/source_manifest.json').read_text())
    target = root/'data/raw'
    target.mkdir(parents=True, exist_ok=True)
    for entry in manifest['datasets']:
        name = entry['filename']
        if Path(name).name != name:
            raise ValueError('Dataset filename must be a basename')
        path = target/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Existing file differs from the reference: {name}. Keep a copy and remove it before downloading again.')
            print(f'Verified {name}', flush=True)
            continue
        if verify_only:
            raise FileNotFoundError(f'Missing {name}; run python download_data.py')
        request = urllib.request.Request(entry['source_url'], headers={'User-Agent': 'stellar-time-series-reproducibility/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f'Download checksum differs: {name}; the upstream file may have changed')
        pending = path.with_suffix(path.suffix+'.download')
        pending.write_bytes(content)
        pending.replace(path)
        print(f'Downloaded and verified {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    acquire(verify_only=args.verify_only)
