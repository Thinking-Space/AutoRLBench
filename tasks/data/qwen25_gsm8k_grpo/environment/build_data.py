"""Materialize the launcher's GSM8K parquet files on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    source = workspace / 'verl/examples/data_preprocess/gsm8k.py'
    output = workspace / 'data/preprocessed'
    evaluation = workspace / 'eval'
    output.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    marker = output / '.complete.json'
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    if marker.is_file():
        recorded = json.loads(marker.read_text())
        if recorded['source_sha256'] == identity and all(
            (workspace / name).is_file() and hashlib.sha256((workspace / name).read_bytes()).hexdigest() == value
            for name, value in recorded['files'].items()
        ):
            print('GSM8K_DATA_READY')
            return
    env = {**os.environ, 'PYTHONPATH': str(workspace / 'verl'), 'HF_HUB_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_DATASETS_CACHE': str(workspace / '.cache/datasets')}
    subprocess.run([sys.executable, str(source), '--local_dataset_path', str(workspace / 'data/gsm8k'),
                    '--local_save_dir', str(output)], cwd=workspace / 'verl', env=env, check=True)
    train = pq.read_table(output / 'train.parquet').to_pylist()
    test = pq.read_table(output / 'test.parquet').to_pylist()
    assert len(train) == 7473 and len(test) == 1319
    assert all(row['prompt'] and row['reward_model']['ground_truth'] is not None for row in train + test)
    order = sorted(range(len(test)), key=lambda i: hashlib.sha256(f'42|validation|{i}'.encode()).digest())
    pq.write_table(pa.Table.from_pylist([test[i] for i in order[:256]]), evaluation / 'validation.parquet')
    pq.write_table(pa.Table.from_pylist(test), evaluation / 'test.parquet')
    files = [output / 'train.parquet', output / 'test.parquet', evaluation / 'validation.parquet', evaluation / 'test.parquet']
    result = {'source_sha256': identity, 'files': {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    temporary = marker.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(marker)
    print('GSM8K_DATA_READY', len(train), len(test))


if __name__ == '__main__':
    main()
