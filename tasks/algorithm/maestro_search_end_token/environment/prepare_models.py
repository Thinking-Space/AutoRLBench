#!/usr/bin/env python3
"""Download all model assets from pinned Hugging Face repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def safe_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'unsafe manifest path: {relative}')
    return root / value


def check_file(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['bytes']:
        raise ValueError(f'missing or wrong-sized model file: {path}')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    if value.hexdigest() != entry['sha256']:
        raise ValueError(f'model checksum mismatch: {path}')


def prepare(workspace, dry_run=False):
    manifest = json.loads((ROOT / 'model_manifest.json').read_text())
    for model in manifest['models']:
        if not re.fullmatch('[0-9a-f]{40}', model['revision']):
            raise ValueError('model revision must be a 40-character commit')
        destination = safe_path(workspace, model['destination'])
        weights = [entry for entry in model['files'] if entry['path'].endswith('.safetensors')]
        if not weights:
            raise ValueError('each model needs pinned weights')
        command = ['autorlbench_fetch_hf', '--type', 'model', '--repo', model['repo'],
                   '--revision', model['revision'], '--dest', str(destination)]
        for entry in model['files']:
            safe_path(destination, entry['path'])
            command += ['--allow', entry['path'], '--expect', entry['path'] + '=' + entry['sha256'],
                        '--expect-size', entry['path'] + '=' + str(entry['bytes'])]
        if dry_run:
            print(json.dumps(command))
            continue
        subprocess.run(command, check=True)
        for entry in model['files']:
            path = safe_path(destination, entry['path'])
            check_file(path, entry)
        print('[prepare] verified model', model['name'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.dry_run)
