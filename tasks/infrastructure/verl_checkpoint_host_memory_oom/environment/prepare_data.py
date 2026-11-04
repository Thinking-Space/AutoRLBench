"""Convert complete official GSM8K splits to verl format; no generated questions."""
import argparse
import hashlib
from pathlib import Path
import re

import pandas as pd

REVISION = '740312add88f781978c0658806c59bc2815b9866'
SOURCES = {
    'train': ('ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d', 7473, 7473),
    'test': ('ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59', 1319, 1319),
}


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    questions = {}
    for split, (digest, total, count) in SOURCES.items():
        path = source / (split + '.parquet')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, split
        frame = pd.read_parquet(path)
        assert len(frame) == total
        records = []
        for index, row in frame.iloc[:count].iterrows():
            answer = re.fullmatch(r'(-?[\d,]+(?:\.\d+)?)', row['answer'].rsplit('####', 1)[1].strip())
            assert answer is not None
            records.append({
                'data_source': 'openai/gsm8k',
                'prompt': [{'role': 'user', 'content': row['question'] + '\nLet\'s think step by step and output the final answer after "####".'}],
                'ability': 'math',
                'reward_model': {'style': 'rule', 'ground_truth': answer[1].replace(',', '')},
                'extra_info': {'index': int(index), 'split': split, 'question': row['question'], 'answer': row['answer']},
            })
        questions[split] = {r['extra_info']['question'] for r in records}
        pd.DataFrame(records).to_parquet(output / (split + '.parquet'), index=False)
    assert not questions['train'] & questions['test']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
