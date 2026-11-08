"""Materialize the retrieval catalog used by this configuration."""
import argparse
import json
import re
from pathlib import Path

def build(rows, passages):
    return list(passages)

def main():
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(root / 'shared/data/train.parquet').to_pylist()
    passages = [json.loads(line) for line in (root / 'shared/data/catalog_002.jsonl').read_text().splitlines()]
    values = build(rows, passages)
    with args.out.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
