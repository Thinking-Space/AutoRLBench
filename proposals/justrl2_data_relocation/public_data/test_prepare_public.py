"""CPU-only tests using four tiny, explicitly synthetic public-style shards."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name('prepare_public.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def storage_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + '\n').encode('utf-8')


class PublicPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='public-math-synthetic-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
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

    def module(self):
        self.assertTrue(SCRIPT.is_file(), 'public store reconstruction implementation is missing')
        spec = importlib.util.spec_from_file_location('public_math_preparer', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def prepare(self, **kwargs):
        return self.module().prepare(self.sources, self.out, selection_path=self.selection_path,
                                     lock_path=self.lock_path, **kwargs)

    def expect_invalid(self, pattern):
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, pattern):
            self.prepare()
        self.assertFalse(self.out.exists(), 'invalid inputs must not create a partial store')

    def update_asset_hash(self, index):
        item = self.lock['files'][index]
        raw = (self.sources / Path(item['path']).name).read_bytes()
        item['size'] = len(raw)
        item['sha256'] = sha(raw)

    def test_exact_runtime_bytes_and_new_public_case(self):
        receipt = self.prepare()
        for name, raw in self.expected.items():
            self.assertEqual((self.out / name).read_bytes(), raw, name)
        files = {str(p.relative_to(self.out)) for p in self.out.rglob('*') if p.is_file()}
        self.assertEqual(files, set(self.expected) | {'CASE.json', 'PUBLIC_PROVENANCE.json'})
        case = json.loads((self.out / 'CASE.json').read_bytes())
        self.assertEqual(case['schema_version'], 1)
        self.assertIs(case['synthetic_fault_injection'], True)
        self.assertEqual(case['difficulty_basis']['kind'], 'frozen_author_supplied')
        self.assertIn('not public dataset measurements', case['difficulty_basis']['description'])
        self.assertEqual(case['source']['repo'], self.lock['repo'])
        self.assertEqual(case['source']['revision'], self.lock['revision'])
        self.assertEqual(case['source']['fileshashes'], {f['path']: f['sha256'] for f in self.lock['files']})
        self.assertNotIn('path', case['source'])
        public_manifest = {key: case['source'][key] for key in ('repo', 'revision', 'config', 'split', 'files')}
        self.assertEqual(case['source'].get('sha256'), sha(canonical(public_manifest)))
        self.assertIn('public source manifest', case['source']['sha256_meaning'])
        self.assertEqual([r['id'] for r in case['rows']], [r['id'] for r in self.selection['rows']])
        for row, selected in zip(case['rows'], self.selection['rows']):
            for key in ('id', 'split', 'difficulty', 'source_line', 'prompt_sha256', 'label_sha256', 'content_sha256'):
                self.assertEqual(row[key], selected[key])
            self.assertEqual(row['public_record_sha256'], selected['public_record_sha256'])
        self.assertEqual(receipt, json.loads((self.out / 'PUBLIC_PROVENANCE.json').read_bytes()))
        self.assertEqual(receipt['runtime_sha256'], self.selection['expected_runtime_sha256'])
        self.assertEqual(receipt['inputs']['selection_sha256'], sha(self.selection_path.read_bytes()))
        self.assertEqual(receipt['inputs']['sources_lock_sha256'], sha(self.lock_path.read_bytes()))
        self.assertEqual(receipt['source'], case['source'])
        self.assertEqual(receipt['case_sha256'], sha((self.out / 'CASE.json').read_bytes()))
        self.assertTrue(receipt['case_is_new_public_provenance'])
        self.assertEqual(receipt['verification']['runtime_files'], 4)
        self.assertEqual(receipt['verification']['selected_pairs'], 4)
        self.assertEqual(receipt['verification']['content_hashes'], 12)

    def test_source_composite_hash_binds_pinned_public_manifest(self):
        self.prepare()
        case = json.loads((self.out / 'CASE.json').read_bytes())
        manifest = {key: self.lock[key] for key in ('repo', 'revision', 'config', 'split')}
        manifest['files'] = [{key: item[key] for key in ('path', 'size', 'sha256', 'rows')}
                             for item in self.lock['files']]
        self.assertEqual(case['source'].get('sha256'), sha(canonical(manifest)))
        self.assertEqual(case['source']['files'], manifest['files'])

    def test_accepts_existing_empty_output_directory(self):
        self.out.mkdir()
        self.prepare()
        self.assertEqual((self.out / 'manifest.json').read_bytes(), self.expected['manifest.json'])

    def test_repeated_output_is_rejected_without_changes(self):
        self.prepare()
        before = {str(p.relative_to(self.out)): p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(FileExistsError, 'absent or empty'):
            self.prepare()
        self.assertEqual(before, {str(p.relative_to(self.out)): p.read_bytes() for p in self.out.rglob('*') if p.is_file()})

    def test_changed_source_size_is_rejected(self):
        p = self.sources / self.lock['files'][0]['path'].split('/')[-1]
        p.write_bytes(p.read_bytes() + b'\n')
        self.expect_invalid('size')

    def test_changed_source_digest_is_rejected(self):
        p = self.sources / self.lock['files'][0]['path'].split('/')[-1]
        p.write_bytes(p.read_bytes().replace(b'Synthetic', b'synthetic'))
        self.expect_invalid('SHA256')

    def test_missing_source_is_rejected_without_output(self):
        (self.sources / self.lock['files'][0]['path'].split('/')[-1]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_insufficient_source_records_are_rejected(self):
        p = self.sources / self.lock['files'][0]['path'].split('/')[-1]
        p.write_bytes(b'\n')
        self.update_asset_hash(0)
        self.expect_invalid('record count')

    def test_repeated_selected_record_is_rejected(self):
        self.selection['rows'][1] = copy.deepcopy(self.selection['rows'][0])
        self.expect_invalid('Duplicate')

    def test_insufficient_selected_records_are_rejected(self):
        self.selection['rows'].pop()
        self.expect_invalid('count')

    def test_public_record_hash_is_verified(self):
        self.selection['rows'][0]['public_record_sha256'] = '0' * 64
        self.expect_invalid('record SHA256')

    def test_prompt_label_and_content_hashes_are_verified(self):
        original = copy.deepcopy(self.selection)
        for field in ('prompt_sha256', 'label_sha256', 'content_sha256'):
            with self.subTest(field=field):
                self.selection = copy.deepcopy(original)
                self.selection['rows'][0][field] = '0' * 64
                self.expect_invalid(field)

    def test_all_runtime_hashes_are_verified(self):
        original = copy.deepcopy(self.selection)
        for name in self.expected:
            with self.subTest(path=name):
                self.selection = copy.deepcopy(original)
                self.selection['expected_runtime_sha256'][name] = '0' * 64
                self.expect_invalid('Runtime SHA256')

    def test_missing_runtime_hash_is_rejected(self):
        self.selection['expected_runtime_sha256'].pop('eval.jsonl')
        self.expect_invalid('runtime file set')

    def test_global_public_order_and_uuid_are_verified(self):
        self.selection['rows'][0]['source_line'] = 99
        self.selection['rows'][0]['id'] = 'easy-00000099'
        self.selection['rows'][0]['uuid'] = 'Math_00099'
        hashes = self.selection['expected_runtime_sha256']
        hashes['live/easy/easy-00000099.json'] = hashes.pop('live/easy/easy-00000001.json')
        self.expect_invalid('global order')

    def test_public_answer_mismatch_is_rejected(self):
        p = self.sources / self.lock['files'][0]['path'].split('/')[-1]
        row = json.loads(p.read_bytes())
        row['ground_truth'] = 'different synthetic answer'
        raw = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
        p.write_bytes(raw)
        self.update_asset_hash(0)
        self.selection['rows'][0]['public_record_sha256'] = sha(raw)
        self.expect_invalid('label_sha256')

    def test_invalid_historical_accuracy_is_rejected(self):
        original = copy.deepcopy(self.selection)
        for accuracy in (True, '0.875', 0.25):
            with self.subTest(accuracy=accuracy):
                self.selection = copy.deepcopy(original)
                self.selection['rows'][0]['historical_accuracy'] = accuracy
                self.expect_invalid('historical_accuracy')

    def test_historical_strata_must_be_declared_author_supplied(self):
        self.selection['historical_strata']['kind'] = 'public_measurement'
        self.expect_invalid('frozen_author_supplied')

    def test_mismatched_public_revision_is_rejected(self):
        self.selection['source']['revision'] = 'b' * 40
        self.expect_invalid('source identity')

    def test_four_shards_are_required(self):
        self.lock['files'].pop()
        self.expect_invalid('four')

    def test_default_selection_and_lock_are_adjacent_to_cli(self):
        self.assertTrue(SCRIPT.is_file(), 'public store reconstruction implementation is missing')
        cli = self.root / 'cli'
        cli.mkdir()
        shutil.copyfile(SCRIPT, cli / 'prepare_public.py')
        shutil.copyfile(self.selection_path, cli / 'selection.json')
        shutil.copyfile(self.lock_path, cli / 'sources.lock.json')
        result = subprocess.run([sys.executable, '-B', str(cli / 'prepare_public.py'),
                                 '--source-dir', str(self.sources), '--out', str(self.out)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / 'eval.jsonl').read_bytes(), self.expected['eval.jsonl'])
        self.assertEqual(json.loads(result.stdout)['runtime_files'], 4)

    def test_optional_download_uses_revision_pins_and_validates_bytes(self):
        module = self.module()
        shutil.rmtree(self.sources)
        requested = []

        def serve(request, timeout):
            url = request.full_url if hasattr(request, 'full_url') else request
            requested.append(url)
            self.assertIn('/resolve/' + self.lock['revision'] + '/data/Math/', url)
            return io.BytesIO(self.public_bytes[url.rsplit('/', 1)[-1]])

        with patch.object(module.urllib.request, 'urlopen', side_effect=serve):
            module.prepare(self.sources, self.out, selection_path=self.selection_path,
                           lock_path=self.lock_path, download=True)
        self.assertEqual(len(requested), 4)
        self.assertEqual((self.out / 'manifest.json').read_bytes(), self.expected['manifest.json'])

    def test_optional_download_rejects_incorrect_bytes(self):
        module = self.module()
        shutil.rmtree(self.sources)
        with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong synthetic bytes')):
            with self.assertRaisesRegex(ValueError, 'size|SHA256'):
                module.prepare(self.sources, self.out, selection_path=self.selection_path,
                               lock_path=self.lock_path, download=True)
        self.assertFalse(self.out.exists())
        self.assertFalse(any(self.sources.glob('*.jsonl')))


if __name__ == '__main__':
    unittest.main()
