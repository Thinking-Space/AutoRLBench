#!/usr/bin/env python3
"""Restore losslessly compressed historical logs in a staged run directory.

Normal prepare restores only workspace/results/recorded_run. Hidden comparison
logs remain in the oracle; reviewers can restore a COPY with --run-dir.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile


def checked_path(root, relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts or len(value.parts) < 2:
        raise ValueError(f'unsafe log path: {relative}')
    if value.parts[0] != 'logs' and value.parts[:2] != ('wandb', 'logs'):
        raise ValueError(f'log path outside historical log directories: {relative}')
    current = root
    for part in value.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'refusing log symlink: {current}')
    return current


def verify(path, expected_bytes, expected_sha256):
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f'log size mismatch or missing file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f'log checksum mismatch: {path}')


def restore(run_dir):
    run_dir = run_dir.resolve()
    manifest_path = run_dir / 'log_archive_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('refusing log manifest symlink')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('invalid historical log archive manifest')
    targets = set()
    for entry in manifest['files']:
        if entry['compressed_path'] != entry['path'] + '.gz' or entry['path'] in targets:
            raise ValueError('invalid or duplicate compressed log path')
        targets.add(entry['path'])
        target = checked_path(run_dir, entry['path'])
        compressed = checked_path(run_dir, entry['compressed_path'])
        if compressed.exists():
            verify(compressed, entry['compressed_bytes'], entry['compressed_sha256'])
        if target.exists():
            # Idempotent, but never silently replace altered historical evidence.
            verify(target, entry['bytes'], entry['sha256'])
        else:
            if not compressed.is_file():
                raise ValueError(f'missing compressed log: {compressed}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as output:
                    temporary = Path(output.name)
                    count = 0
                    with gzip.open(compressed, 'rb') as stream:
                        while True:
                            block = stream.read(min(1024 * 1024, entry['bytes'] - count + 1))
                            if not block:
                                break
                            count += len(block)
                            if count > entry['bytes']:
                                raise ValueError(f'decompressed log exceeds declared size: {target}')
                            output.write(block)
                verify(temporary, entry['bytes'], entry['sha256'])
                temporary.chmod(0o644)
                temporary.replace(target)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        # Only the staged copy is modified by prepare; the submission keeps .gz.
        if compressed.exists():
            compressed.unlink()
    print(f'[prepare] Restored and verified {len(targets)} historical logs: {run_dir}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--workspace', type=Path, help='staged agent-visible workspace')
    group.add_argument('--run-dir', type=Path, help='explicit copied historical run for reviewer checks')
    args = parser.parse_args()
    restore(args.run_dir if args.run_dir is not None else args.workspace / 'results/recorded_run')
