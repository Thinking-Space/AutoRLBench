"""Offline integrity tests for the preparation artifacts; no model or network."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import (POSITION_COLUMN, SOURCE_ROW_COLUMN, data_manifest,
                          derive_revision, digest, encoded, seed_metadata,
                          verify_outputs)


def description(path, rows=None):
    value = {'sha256': digest(path), 'size': path.stat().st_size}
    if rows is not None:
        value['rows'] = rows
    return value


def fixture_assets(root):
    base = root / 'base'
    expected = root / 'expected'
    base.mkdir(); expected.mkdir()
    records = [
        {'prompt': [{'role': 'user', 'content': 'Compute 2+2.'}], 'answer': '4'},
        {'prompt': [{'role': 'user', 'content': 'Compute 3+4.'}], 'answer': '7'},
        {'prompt': [{'role': 'user', 'content': 'Compute 2+2.'}], 'answer': '4'},
        {'prompt': [{'role': 'user', 'content': 'Compute 2+2.'}], 'answer': '5'},
        {'prompt': [{'role': 'user', 'content': 'Compute 9-3.'}], 'answer': '6'},
    ]
    source = pa.Table.from_pylist(records)
    validation = pa.Table.from_pylist([records[-1]])
    pq.write_table(source, base / 'train.parquet', compression='zstd')
    pq.write_table(validation, base / 'val.parquet', compression='zstd')
    source = pq.read_table(base / 'train.parquet')
    registry = {
        'base_derived': {name: description(base / name, rows)
                         for name, rows in (('train.parquet', 5), ('val.parquet', 1))},
        'revision_seed': 20261104,
        'processing_contract': {'version': 1, 'stage': 'structured_messages_before_tokenization',
                                'prompt_key': 'prompt', 'normalization': 'identity'},
        'derived': {},
    }
    permutation = np.random.default_rng(20261104).permutation(5)
    # Construct expected complete rows from Python objects, independently of
    # the Arrow take/append operations used by the actual generator.
    revised = pa.Table.from_pylist([records[int(i)] for i in permutation], schema=source.schema)
    revised = revised.append_column(POSITION_COLUMN, pa.array(list(range(5)), type=pa.int64()))
    revised = revised.append_column(SOURCE_ROW_COLUMN, pa.array(permutation, type=pa.int64()))
    pq.write_table(revised, expected / 'train.parquet', compression='zstd')
    pq.write_table(source.select(['prompt']), expected / 'prompt-cache-v1.parquet', compression='zstd')
    (expected / 'val.parquet').write_bytes((base / 'val.parquet').read_bytes())
    for name, rows in (('train.parquet', 5), ('val.parquet', 1), ('prompt-cache-v1.parquet', 5)):
        registry['derived'][name] = description(expected / name, rows)
    (expected / 'prompt-cache-v1.json').write_bytes(encoded(seed_metadata(registry)))
    registry['derived']['prompt-cache-v1.json'] = description(expected / 'prompt-cache-v1.json')
    manifest_raw = encoded(data_manifest(registry))
    registry['prepared_manifest'] = {'sha256': hashlib.sha256(manifest_raw).hexdigest(),
                                     'size': len(manifest_raw)}
    return base, registry, records


def test_complete_rows_duplicates_and_validation_bytes_are_retained(tmp_path):
    base, registry, records = fixture_assets(tmp_path)
    out = tmp_path / 'data'
    assert derive_revision(base / 'train.parquet', base / 'val.parquet', out, registry)['status'] == 'prepared'
    actual = pq.read_table(out / 'train.parquet').to_pylist()
    raw_rows = [{key: value for key, value in row.items()
                 if key not in (POSITION_COLUMN, SOURCE_ROW_COLUMN)} for row in actual]
    assert Counter(json.dumps(row, sort_keys=True) for row in raw_rows) == Counter(json.dumps(row, sort_keys=True) for row in records)
    for position, row in enumerate(actual):
        assert row[POSITION_COLUMN] == position
        assert raw_rows[position] == records[row[SOURCE_ROW_COLUMN]]
    assert (out / 'val.parquet').read_bytes() == (base / 'val.parquet').read_bytes()
    assert set(json.loads((out / 'manifest.json').read_text())) == {'files', 'rows', 'processing_contract'}
    assert derive_revision(base / 'train.parquet', base / 'val.parquet', out, registry)['status'] == 'reused'


def test_each_reuse_rechecks_content_and_marker(tmp_path):
    base, registry, _ = fixture_assets(tmp_path)
    out = tmp_path / 'data'
    derive_revision(base / 'train.parquet', base / 'val.parquet', out, registry)
    target = out / 'prompt-cache-v1.parquet'
    target.chmod(0o644)
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match='Asset identity mismatch'):
        derive_revision(base / 'train.parquet', base / 'val.parquet', out, registry)
    target.write_bytes(raw)
    (out / '.complete').write_text('{}')
    with pytest.raises(ValueError, match='completion marker differs'):
        verify_outputs(out, registry)


def test_partial_output_and_modified_source_are_not_overwritten(tmp_path):
    base, registry, _ = fixture_assets(tmp_path)
    out = tmp_path / 'data'
    out.mkdir(); (out / 'partial').write_text('partial')
    with pytest.raises(ValueError, match='Nonempty data directory'):
        derive_revision(base / 'train.parquet', base / 'val.parquet', out, registry)
    registry = copy.deepcopy(registry)
    registry['base_derived']['train.parquet']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Asset identity mismatch'):
        derive_revision(base / 'train.parquet', base / 'val.parquet', tmp_path / 'other', registry)
