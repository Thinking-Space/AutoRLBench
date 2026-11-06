#!/usr/bin/env python3
"""Verify downloaded data, frozen tool-result inputs and model assets."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(root, entry):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'unsafe asset path: {relative}')
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing/non-regular asset: {path}')
    if path.stat().st_size != entry['bytes']:
        raise ValueError(f'asset size mismatch: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != entry['sha256']:
        raise ValueError(f'asset checksum mismatch: {path}')


def check(workspace, data_only=False):
    data = json.loads((ROOT / 'data_manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda entry: verify(workspace, entry), data['files']))
    print('[prepare] verified data and frozen tool-input files:', len(data['files']), flush=True)
    if not data_only:
        manifest = json.loads((ROOT / 'model_manifest.json').read_text())
        for model in manifest['models']:
            for entry in model['files']:
                verify(workspace / model['destination'], entry)
            print('[prepare] verified checkpoint:', model['name'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    check(args.workspace.resolve(), args.data_only)
