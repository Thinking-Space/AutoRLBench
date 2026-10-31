#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned public source, exact CPU safetensors, and the public math store.

Example: python3 prepare_assets.py --out /work/prepared --download
Requires torch and safetensors only for the CPU tensor conversion. No GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
ARCHIVE = 'source.tar.gz'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    return {'size': Path(path).stat().st_size, 'sha256': file_sha(path)}


def safe_path(root, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or str(relative) != name:
        raise ValueError('Invalid inventory path: ' + name)
    path = Path(root) / name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Inventory path escapes root: ' + name)
    return path


def verify_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing regular asset: ' + str(path))
    if path.stat().st_size != expected['size']:
        raise ValueError('Asset size mismatch: ' + str(path))
    if file_sha(path) != expected['sha256']:
        raise ValueError('Asset SHA256 hash mismatch: ' + str(path))


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Missing regular directory: ' + str(root))
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Non-regular inventory entry: ' + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_record(path)
    return result


def verify_inventory(root, expected):
    for name in expected:
        safe_path(root, name)
    actual = inventory(root)
    if actual != expected:
        raise ValueError('File inventory hash/size mismatch: ' + str(root))
    return actual


def fetch_asset(target, pin, local=None):
    """Stream one pinned asset; publish only verified bytes, never trust a cache hit."""
    target = Path(target)
    if target.exists() or target.is_symlink():
        verify_file(target, pin)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    try:
        with temporary.open('xb') as output:
            if local is not None:
                verify_file(local, pin)
                with Path(local).open('rb') as source:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
            else:
                request = urllib.request.Request(pin['url'], headers={'User-Agent': 'AutoRLBench-public-runtime/1'})
                with urllib.request.urlopen(request, timeout=120) as source:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
        verify_file(temporary, pin)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def verify_source_archive(path, assets, pins):
    verify_file(path, assets['public_source_archive'])
    prefix = 'JustRL-II-' + assets['justrl2_revision']
    actual = {}
    with tarfile.open(path) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != prefix or '..' in parts or member.name.startswith('/'):
                raise ValueError('Unexpected source archive path')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Source archive must contain regular files')
            name = '/'.join(parts[1:])
            if not name or name in actual:
                raise ValueError('Duplicate/invalid source archive file')
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    digest.update(chunk)
            actual[name] = digest.hexdigest()
    if actual != pins:
        raise ValueError('Source archive file inventory differs from pinned revision')


def verify_converter_input(model, receipt, assets):
    files = verify_inventory(model, receipt['files'])
    metadata = {Path(a['path']).name: {'size': a['size'], 'sha256': a['sha256']}
                for a in assets['assets'] if not a['path'].endswith('.bin')}
    if set(files) != set(metadata) | {'model.safetensors'} or any(files[k] != v for k, v in metadata.items()):
        raise ValueError('Converter tokenizer/config files differ from public pins')
    tensor = receipt['tensor_check']
    original = next(a['sha256'] for a in assets['assets'] if a['path'] == 'model/pytorch_model.bin')
    target = files['model.safetensors']
    if (receipt['format'] != 'losslessly_reencoded_safetensors' or
            tensor['source_sha256'] != original or tensor['target_sha256'] != target['sha256'] or
            tensor['target_size'] != target['size'] or tensor['all_tensors_exactly_equal'] is not True or
            type(tensor['tensor_count']) is not int or tensor['tensor_count'] < 1):
        raise ValueError('Missing exact public tensor conversion evidence')
    return files


def verify_dataset(dataset):
    """Bind the frozen runtime and new CASE to the public preparer's own receipts."""
    public = HERE / 'public_data'
    receipt = read_json(dataset / 'PUBLIC_PROVENANCE.json')
    selection = read_json(public / 'selection.json')
    lock = read_json(public / 'sources.lock.json')
    for key, filename in (('selection_sha256', 'selection.json'), ('sources_lock_sha256', 'sources.lock.json')):
        if receipt['inputs'][key] != file_sha(public / filename):
            raise ValueError('Public dataset preparation input changed: ' + filename)
    if receipt['preparer']['sha256'] != file_sha(public / 'prepare_public.py'):
        raise ValueError('Public dataset preparer changed')
    if receipt['runtime_sha256'] != selection['expected_runtime_sha256']:
        raise ValueError('Public runtime files differ from frozen selection')
    expected = {name: {'sha256': digest, 'size': receipt['runtime_sizes'][name]}
                for name, digest in receipt['runtime_sha256'].items()}
    expected['CASE.json'] = {'sha256': receipt['case_sha256'], 'size': (dataset / 'CASE.json').stat().st_size}
    expected['PUBLIC_PROVENANCE.json'] = file_record(dataset / 'PUBLIC_PROVENANCE.json')
    verify_inventory(dataset, expected)
    case = read_json(dataset / 'CASE.json')
    source = case['source']
    public_files = [{key: item[key] for key in ('path', 'size', 'sha256', 'rows')} for item in lock['files']]
    if (source != receipt['source'] or source['files'] != public_files or
            any(source[k] != lock[k] for k in ('repo', 'revision', 'config', 'split')) or
            case['synthetic_fault_injection'] is not True or receipt['case_is_new_public_provenance'] is not True):
        raise ValueError('CASE must describe the pinned public source')
    manifest = {key: source[key] for key in ('repo', 'revision', 'config', 'split', 'files')}
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if source.get('sha256') != digest:
        raise ValueError('Public CASE source composite SHA256 differs from the pinned public manifest')
    return expected


def verify_prepared(root):
    root = Path(root)
    assets = read_json(HERE / 'assets.json')
    pins = read_json(HERE / 'runtime_support/source_files.json')['files']
    verify_source_archive(root / ARCHIVE, assets, pins)
    raw = {Path(a['path']).name: {'size': a['size'], 'sha256': a['sha256']} for a in assets['assets']}
    verify_inventory(root / 'model', raw)
    converter = read_json(root / 'safetensors_receipt.json')
    verify_converter_input(root / 'hf-safetensors', converter, assets)
    verify_dataset(root / 'dataset')
    receipt = read_json(root / 'prepared_receipt.json')
    if receipt['model_revision'] != assets['model_revision'] or receipt['source_revision'] != assets['justrl2_revision']:
        raise ValueError('Prepared revisions differ from public pins')
    if receipt['assets_lock_sha256'] != file_sha(HERE / 'assets.json'):
        raise ValueError('Prepared asset lock changed')
    for name, expected in receipt['receipts'].items():
        verify_file(safe_path(root, name), expected)
    return assets, converter


def prepare(args):
    root = args.out.resolve()
    if root.exists():
        raise FileExistsError('Use a new prepared directory: ' + str(root))
    assets = read_json(HERE / 'assets.json')
    root.mkdir(parents=True)
    fetch_asset(root / ARCHIVE, assets['public_source_archive'],
                args.asset_dir / ARCHIVE if args.asset_dir else None)
    verify_source_archive(root / ARCHIVE, assets, read_json(HERE / 'runtime_support/source_files.json')['files'])
    for asset in assets['assets']:
        print('Preparing ' + asset['path'], flush=True)
        fetch_asset(safe_path(root, asset['path']), asset,
                    safe_path(args.asset_dir, asset['path']) if args.asset_dir else None)
    derived = root / 'hf-safetensors'
    derived.mkdir()
    for asset in assets['assets']:
        if not asset['path'].endswith('.bin'):
            shutil.copyfile(root / asset['path'], derived / Path(asset['path']).name)
    from runtime_support.checkpoint_evidence import convert_bin_to_safetensors
    tensor_check = convert_bin_to_safetensors(root / 'model/pytorch_model.bin', derived / 'model.safetensors')
    converter = {'format': 'losslessly_reencoded_safetensors', 'tensor_check': tensor_check, 'files': inventory(derived)}
    verify_converter_input(derived, converter, assets)
    write_json(root / 'safetensors_receipt.json', converter)
    command = [sys.executable, str(HERE / 'public_data/prepare_public.py'),
               '--source-dir', str(args.source_dir.resolve() if args.source_dir else root / 'public-source'),
               '--out', str(root / 'dataset')]
    if args.download:
        command.append('--download')
    subprocess.run(command, check=True)
    verify_dataset(root / 'dataset')
    write_json(root / 'prepared_receipt.json', {
        'schema_version': 1, 'model_revision': assets['model_revision'], 'source_revision': assets['justrl2_revision'],
        'assets_lock_sha256': file_sha(HERE / 'assets.json'),
        'receipts': {name: file_record(root / name) for name in
                     ('safetensors_receipt.json', 'dataset/PUBLIC_PROVENANCE.json', 'dataset/CASE.json')},
    })
    verify_prepared(root)
    print('PREPARED_ROOT=' + str(root), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New prepared asset root')
    parser.add_argument('--source-dir', type=Path, help='Local directory containing four pinned public Math shards')
    parser.add_argument('--download', action='store_true', help='Permit downloading missing public Math shards')
    parser.add_argument('--asset-dir', type=Path, help='Offline cache with source.tar.gz and model/* matching assets.json')
    args = parser.parse_args(argv)
    if not args.source_dir and not args.download:
        parser.error('provide --source-dir or --download for the public Math shards')
    try:
        prepare(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
