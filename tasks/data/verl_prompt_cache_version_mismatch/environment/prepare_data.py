#!/usr/bin/env python3
"""Build the pinned data revision and its immutable processed-message seed.

All original source rows are retained. Preparation verifies input/output bytes
on both first use and reuse; it does not inspect model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare_base_data


POSITION_COLUMN = 'cache_version_position'
SOURCE_ROW_COLUMN = 'cache_version_source_row'


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check(path, specification):
    path = Path(path)
    if (not path.is_file() or path.stat().st_size != specification['size']
            or digest(path) != specification['sha256']):
        raise ValueError(f'Asset identity mismatch: {path.name}')
    if 'rows' in specification and pq.ParquetFile(path).metadata.num_rows != specification['rows']:
        raise ValueError(f'Asset row count differs: {path.name}')


def seed_metadata(registry):
    cache = registry['derived']['prompt-cache-v1.parquet']
    source = registry['base_derived']['train.parquet']
    return {
        'identity': {'source_key': 'train.parquet', 'rows': source['rows'],
                     'source_sha256': source['sha256'],
                     'preprocessing': registry['processing_contract']},
        'artifact': {'path': 'prompt-cache-v1.parquet', 'size_bytes': cache['size'],
                     'sha256': cache['sha256']},
    }


def data_manifest(registry):
    return {
        'files': {name: {'path': name, 'size_bytes': value['size'],
                         'sha256': value['sha256']}
                  for name, value in registry['derived'].items()},
        'rows': {'train': registry['derived']['train.parquet']['rows'],
                 'val': registry['derived']['val.parquet']['rows']},
        'processing_contract': registry['processing_contract'],
    }


def verify_outputs(output, registry):
    output = Path(output)
    for name, specification in registry['derived'].items():
        check(output / name, specification)
    check(output / 'manifest.json', registry['prepared_manifest'])
    if (output / 'manifest.json').read_bytes() != encoded(data_manifest(registry)):
        raise ValueError('Prepared manifest differs from the pinned registry')
    if (output / 'prompt-cache-v1.json').read_bytes() != encoded(seed_metadata(registry)):
        raise ValueError('Processed-message metadata differs from the pinned registry')
    marker = json.loads((output / '.complete').read_text())
    if marker != {'manifest_sha256': registry['prepared_manifest']['sha256']}:
        raise ValueError('Preparation completion marker differs')


def derive_revision(source_train, source_val, output, registry):
    """Construct whole-row ordering and pre-tokenization message artifacts."""
    source_train, source_val, output = map(Path, (source_train, source_val, output))
    for name, path in (('train.parquet', source_train), ('val.parquet', source_val)):
        check(path, registry['base_derived'][name])
    if (output / '.complete').exists():
        verify_outputs(output, registry)
        return {'status': 'reused', 'verified': True}
    if output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty data directory without completion marker; do not overwrite')
    source = pq.read_table(source_train)
    if any(name in source.column_names for name in (POSITION_COLUMN, SOURCE_ROW_COLUMN)):
        raise ValueError('Source already has positional revision columns')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.data-preparing-') as temporary:
        stage = Path(temporary)
        size = source.num_rows
        permutation = np.random.default_rng(registry['revision_seed']).permutation(size)
        pq.write_table(source.select(['prompt']), stage / 'prompt-cache-v1.parquet', compression='zstd')
        revised = source.take(pa.array(permutation, type=pa.int64()))
        revised = revised.append_column(POSITION_COLUMN, pa.array(np.arange(size), type=pa.int64()))
        revised = revised.append_column(SOURCE_ROW_COLUMN, pa.array(permutation, type=pa.int64()))
        pq.write_table(revised, stage / 'train.parquet', compression='zstd')
        shutil.copyfile(source_val, stage / 'val.parquet')
        (stage / 'prompt-cache-v1.json').write_bytes(encoded(seed_metadata(registry)))
        (stage / 'manifest.json').write_bytes(encoded(data_manifest(registry)))
        for name, specification in registry['derived'].items():
            check(stage / name, specification)
        check(stage / 'manifest.json', registry['prepared_manifest'])
        output.mkdir(parents=True, exist_ok=True)
        for name in (*registry['derived'], 'manifest.json'):
            os.replace(stage / name, output / name)
            (output / name).chmod(0o444)
        marker = {'manifest_sha256': registry['prepared_manifest']['sha256']}
        (output / '.complete').write_bytes(encoded(marker))
    verify_outputs(output, registry)
    return {'status': 'prepared', 'verified': True}


def run(workspace, registry_path):
    workspace, registry_path = Path(workspace).resolve(), Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text())
    prepare_base_data.run(workspace, registry_path)
    base = workspace / 'assets/base-data'
    return derive_revision(base / 'train.parquet', base / 'val.parquet',
                           workspace / 'data', registry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        default=Path(os.environ.get('AUTORLBENCH_WORKSPACE', '/workspace')))
    parser.add_argument('--registry', type=Path,
                        default=Path(__file__).with_name('asset_registry.json'))
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.workspace, arguments.registry), indent=2))
