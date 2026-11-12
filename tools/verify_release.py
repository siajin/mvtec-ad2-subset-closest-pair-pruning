"""Verify public payload and derivatives against their declared SHA256."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest_path = ROOT/'provenance/RELEASE_SHA256.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['files']:
        p = ROOT/row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    source = json.loads((ROOT/'provenance/SOURCE_ARTIFACTS.json').read_text(encoding='utf-8'))
    for row in source['artifacts']:
        if 'public_copy' in row:
            assert sha(ROOT/row['public_copy']) == row['sha256'], row['public_copy']
        if 'public_derivative' in row:
            assert sha(ROOT/row['public_derivative']) == row['published_sha256'], row['public_derivative']
    print(f"Verified {len(manifest['files'])} release files and source-linked derivatives.")


if __name__ == '__main__':
    main()
