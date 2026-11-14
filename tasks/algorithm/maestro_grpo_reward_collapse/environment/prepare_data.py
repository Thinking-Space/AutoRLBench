#!/usr/bin/env python3
"""Fetch the pinned Maestro dataset; verify every file before model downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from check_assets import check, verify
from prepare_models import safe_path

ROOT = Path(__file__).resolve().parent


def download_command(workspace, manifest):
    dataset = manifest['download']
    if not re.fullmatch('[0-9a-f]{40}', dataset['revision']):
        raise ValueError('dataset revision must be a 40-character commit')
    destination = safe_path(workspace, dataset['destination'])
    command = [sys.executable, str(ROOT / 'fetch_hf.py'), '--type', 'dataset',
               '--repo', dataset['repo'],
               '--revision', dataset['revision'], '--dest', str(destination)]
    for pattern in dataset['allow']:
        safe_path(destination, pattern)
        command += ['--allow', pattern]
    # The 19k image bytes are verified after materialization by check_assets
    # (manifest['files'] sha256s), not at fetch time: the task-local fetch_hf
    # helper does not checksum, and argv stays far below ARG_MAX.
    if not any(entry['path'].endswith('.parquet') for entry in manifest['files']):
        raise ValueError('dataset download needs pinned parquet checksums')
    return command


def prepare(workspace, dry_run=False):
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    command = download_command(workspace, manifest)
    if dry_run:
        print(json.dumps(command))
        return
    if manifest['download']['status'] != 'ready':
        raise RuntimeError(
            'locaszzz/maestro_data has not yet been pinned with all required data '
            'and tool-result files. Verify the uploaded repository contents with '
            'environment/pin_dataset.py --revision <uploaded-commit>. '
            'No models were downloaded and no local-data fallback was used.')
    subprocess.run(command, check=True)
    # The HF repository has tool_results/ at its root. Materialize the snapshot
    # only into data/, never results/: the shared fetch helper replaces --dest.
    # Move the three verified JSONs individually, preserving recorded_run/ and
    # all other results. Repeating prepare replaces only these declared assets.
    mappings = manifest['download'].get('file_mappings', {})
    for entry in manifest['files']:
        if entry['path'] not in mappings:
            continue
        remote = mappings[entry['path']]
        downloaded_root = safe_path(workspace, manifest['download']['destination'])
        source = safe_path(downloaded_root, remote)
        destination = safe_path(workspace, entry['path'])
        verify(downloaded_root, dict(entry, path=remote))
        if destination.is_symlink():
            raise ValueError(f'refusing tool-result symlink: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        if not any(source.parent.iterdir()):
            source.parent.rmdir()
    check(workspace, data_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.dry_run)
