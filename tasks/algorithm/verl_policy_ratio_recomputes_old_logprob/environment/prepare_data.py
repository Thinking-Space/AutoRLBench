"""Pinned official GSM8K records; retain original questions and worked answers."""
import hashlib
import json
from pathlib import Path
import re
import random

try:
    from .profiles import experiment_profile
except ImportError:
    from profiles import experiment_profile

REVISION = '740312add88f781978c0658806c59bc2815b9866'
SOURCES = {
    'train': ('ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d', 7473, 256),
    'test': ('ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59', 1319, 64),
}
INSTRUCTION = ('Solve the following grade-school math word problem. Show your reasoning '
               'and put your final numerical answer inside \\boxed{}.\n\n')


def build(source, output, kind):
    import pandas as pd
    output.mkdir(parents=True, exist_ok=True)
    corpus, files, questions = {}, [], {}
    profile = experiment_profile(kind)
    for split, (digest, total, count) in SOURCES.items():
        path = Path(source)/(split+'.parquet')
        if not path.is_file():
            path = Path(source)/'main'/(split+'-00000-of-00001.parquet')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, split
        frame = pd.read_parquet(path)
        assert len(frame) == total, split
        count = profile['held_out_rows'] if split == 'test' else profile['train_rows']
        indices = (sorted(random.Random(profile['validation_sample_seed']).sample(range(total), count))
                   if split == 'test' and profile['validation_sample_seed'] is not None else list(range(count)))
        records = []
        corpus[split] = {}
        for index, row in frame.iloc[indices].iterrows():
            answer = row['answer'].rsplit('####', 1)[1].strip().replace(',', '')
            assert re.fullmatch(r'-?\d+(?:\.\d+)?', answer), answer
            corpus[split][str(index)] = row['question']
            content = INSTRUCTION+row['question']
            if kind == 'environment':
                content = f'[gsm8k:{split}:{index}]\n'+INSTRUCTION+row['question']
            records.append({'data_source': 'openai/gsm8k',
                'prompt': [{'role': 'user', 'content': content}], 'ability': 'math',
                'reward_model': {'style': 'rule', 'ground_truth': answer},
                'extra_info': {'index': int(index), 'split': split,
                    'question': row['question'], 'answer': row['answer'], 'canonical_answer': answer}})
        pd.DataFrame(records).to_parquet(output/(split+'.parquet'), index=False)
        (output/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        questions[split] = {r['extra_info']['question'] for r in records}
        files.append({'split': split, 'source_indices': indices, 'rows': count,
                      'path': split+'.parquet', 'sha256': hashlib.sha256((output/(split+'.parquet')).read_bytes()).hexdigest()})
    assert not questions['train'] & questions['test'], 'train/test questions overlap'
    # Question-only provenance corpus; no answers or solutions.
    (output/'question_corpus.json').write_text(json.dumps(corpus, indent=2)+'\n')
    manifest = {'source': 'openai/gsm8k', 'config': 'main', 'revision': REVISION,
        'selection': 'First 256 official train rows and first 64 official test rows; no generated examples.',
        'upstream': [{'path': 'main/'+s+'-00000-of-00001.parquet', 'sha256': v[0], 'rows': v[1]} for s,v in SOURCES.items()],
        'preprocessing': 'Original question and worked answer retained in extra_info; standard answer-format instruction; answer extracted after ####.',
        'files': files}
    if kind == 'algorithm':
        manifest['selection'] = 'First 256 official train rows; 100 official test rows sampled without replacement using Python random.Random(42), sorted by source index; no generated examples.'
        manifest['validation_sample_seed'] = profile['validation_sample_seed']
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', required=True)
    args = parser.parse_args()
    build(args.source, args.output, args.kind)
