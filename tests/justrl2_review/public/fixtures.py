"""Four tiny public-style shards with exact source/content pins; no test functions."""
import hashlib
import json
from pathlib import Path
from tests.justrl2_review.fixtures import FAULTY, load


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def storage_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


class PublicFixture:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True)
        self.sources = self.root / 'sources'
        self.sources.mkdir()
        self.out = self.root / 'store'
        self.lock_path = self.root / 'sources.lock.json'
        self.selection_path = self.root / 'selection.json'
        source = {'repo': 'synthetic/example-math', 'revision': 'a' * 40,
                  'config': 'Math', 'split': 'train'}
        self.lock = dict(schema_version=1, **source, files=[])
        self.selection = {
            'schema_version': 1, 'source': source,
            'historical_strata': {
                'kind': 'frozen_author_supplied',
                'description': 'Synthetic frozen author labels; not public dataset measurements.',
                'easy_min': 0.875, 'hard_max': 0.125,
            },
            'expected_counts': {'train': {'easy': 1, 'hard': 1}, 'eval': {'easy': 1, 'hard': 1}},
            'rows': [], 'expected_runtime_sha256': {},
        }
        self.public_bytes = {}
        self.payloads = []
        self.expected = {}
        manifest = []
        eval_lines = []
        for index, (difficulty, split, accuracy) in enumerate([
            ('easy', 'train', 0.875), ('hard', 'train', 0.125),
            ('easy', 'eval', 1.0), ('hard', 'eval', 0.0),
        ], 1):
            public = {'uuid': 'Math_%05d' % index, 'query': 'Synthetic arithmetic α %d?\nAnswer in \\boxed{}.' % index,
                      'ground_truth': str(index), 'source': 'synthetic-fixture', 'domain': 'Math'}
            raw = (json.dumps(public, ensure_ascii=False) + '\n').encode('utf-8')
            name = 'Math_part-%d-of-4.jsonl' % index
            self.public_bytes[name] = raw
            (self.sources / name).write_bytes(raw)
            file = 'data/Math/' + name
            self.lock['files'].append({'path': file, 'size': len(raw), 'sha256': sha(raw), 'rows': 1})
            identifier = '%s-%08d' % (difficulty, index)
            payload = {'id': identifier,
                       'prompt': [{'role': 'user', 'content': public['query']}],
                       'label': public['ground_truth'],
                       'metadata': {'difficulty': difficulty, 'source_line': index,
                                    'historical_accuracy': accuracy}}
            self.payloads.append(payload)
            content = {'prompt': payload['prompt'], 'label': payload['label']}
            selected = {'id': identifier, 'uuid': public['uuid'], 'split': split,
                        'difficulty': difficulty, 'source_line': index, 'historical_accuracy': accuracy,
                        'file': file, 'row_index_0based': 0, 'public_record_sha256': sha(raw),
                        'prompt_sha256': sha(canonical(payload['prompt'])),
                        'label_sha256': sha(canonical(payload['label'])),
                        'content_sha256': sha(canonical(content))}
            self.selection['rows'].append(selected)
            if split == 'train':
                path = 'live/%s/%s.json' % (difficulty, identifier)
                self.expected[path] = storage_bytes(payload)
                manifest.append({'id': identifier, 'difficulty': difficulty, 'path': path,
                                 'sha256': sha(self.expected[path]), 'source_line': index, 'split': split})
            else:
                eval_lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        self.expected['manifest.json'] = storage_bytes({'schema_version': 1, 'rows': manifest})
        self.expected['eval.jsonl'] = ''.join(eval_lines).encode('utf-8')
        self.selection['expected_runtime_sha256'] = {name: sha(raw) for name, raw in self.expected.items()}
        self.save_inputs()

    def save_inputs(self):
        self.lock_path.write_bytes(storage_bytes(self.lock))
        self.selection_path.write_bytes(storage_bytes(self.selection))

    def prepare(self):
        script = FAULTY / 'environment/asset_tools/public_data/prepare_public.py'
        return load(script).prepare(self.sources, self.out, selection_path=self.selection_path,
                                    lock_path=self.lock_path)
