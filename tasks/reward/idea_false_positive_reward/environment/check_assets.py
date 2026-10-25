#!/usr/bin/env python3
"""Read-only verification of prepared dataset inputs and bundled evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def inside(root, relative):
    root = root.resolve()
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise RuntimeError('Asset must stay inside workspace')
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise RuntimeError('Asset must stay inside workspace')
    return path


def verify_asset(root, entry):
    """Read-only size/hash validation of an existing original file."""
    if 'storage' in entry:
        raise RuntimeError('Archive storage is not supported in this unpacked package')
    path = inside(root, entry['path'])
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            if size > entry['bytes']:
                raise RuntimeError('Asset size mismatch: ' + entry['path'])
            digest.update(chunk)
    if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise RuntimeError('Asset size/hash mismatch: ' + entry['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true',
                        help='Compatibility alias: every invocation is read-only validation')
    args = parser.parse_args()
    root = args.workspace.resolve()
    manifest = json.loads(args.manifest.read_text())
    if 'workspace_archives' in manifest:
        raise RuntimeError('Unpacked manifest must use workspace_files, not workspace_archives')
    for entry in manifest['files'] + manifest.get('workspace_files', []):
        verify_asset(root, entry)
    runtime = args.manifest.parent / 'runtime'
    lock = json.loads((runtime / 'runtime-lock.json').read_text())
    for record in lock['environments'].values():
        path = runtime / record['requirements']
        if path.is_symlink() or path.parent != runtime:
            raise RuntimeError('Runtime requirements must stay inside runtime/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['requirements_sha256']:
            raise RuntimeError('Runtime requirements hash mismatch')
    print(json.dumps({'dataset_assets_verified': len(manifest['files']),
                      'workspace_files_verified': len(manifest.get('workspace_files', [])),
                      'runtime_locks_verified': len(lock['environments']),
                      'read_only': True, 'decompression_required': False,
                      'model_download_required': False, 'network_required': False}))


if __name__ == '__main__':
    main()
