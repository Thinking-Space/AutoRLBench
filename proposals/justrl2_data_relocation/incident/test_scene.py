"""Stdlib schema/filesystem tests with explicitly SYNTHETIC, unloadable checkpoints.

These fixtures are not training evidence and never establish a model restore.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('scene.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def inventory(root):
    return {p.relative_to(root).as_posix(): {'size': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


class SceneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='synthetic-scene-schema-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefix = self.root / 'prefix'
        self.scene = self.root / 'scene'
        self.selection_path = self.root / 'selection.json'
        self.dataset = self.prefix / 'dataset'
        self.rows, case_rows, eval_rows = [], [], []
        for index, (split, difficulty) in enumerate([
                ('train', 'easy'), ('train', 'hard'), ('eval', 'easy'), ('eval', 'hard')], 1):
            identifier = f'{difficulty}-{index:08d}'
            payload = {'id': identifier, 'prompt': [{'role': 'user', 'content': f'Synthetic {index}+0?'}],
                       'label': str(index), 'metadata': {'difficulty': difficulty,
                       'source_line': index, 'historical_accuracy': 1.0 if difficulty == 'easy' else 0.0}}
            selected = {'id': identifier, 'difficulty': difficulty, 'split': split,
                        'source_line': index, 'uuid': f'SYNTHETIC_{index}',
                        'file': 'synthetic.jsonl', 'row_index_0based': index - 1,
                        'public_record_sha256': canonical_sha(payload),
                        'historical_accuracy': payload['metadata']['historical_accuracy'],
                        'prompt_sha256': canonical_sha(payload['prompt']),
                        'label_sha256': canonical_sha(payload['label']),
                        'content_sha256': canonical_sha({k: payload[k] for k in ('prompt', 'label')})}
            case_rows.append(dict(selected, source_row_id=selected['uuid'], public_file=selected['file']))
            if split == 'train':
                name = f'live/{difficulty}/{identifier}.json'
                write_json(self.dataset / name, payload)
                self.rows.append({'id': identifier, 'difficulty': difficulty, 'split': split,
                                  'source_line': index, 'path': name, 'sha256': sha(self.dataset / name)})
            else:
                eval_rows.append(payload)
        write_json(self.dataset / 'manifest.json', {'schema_version': 1, 'rows': self.rows})
        write_lines(self.dataset / 'eval.jsonl', eval_rows)
        runtime = inventory(self.dataset)
        source = {'repo': 'synthetic/schema-only', 'revision': 'a' * 40, 'config': 'Math', 'split': 'train'}
        counts = {'train': {'easy': 1, 'hard': 1}, 'eval': {'easy': 1, 'hard': 1}}
        self.selection = {'schema_version': 1, 'source': source, 'rows': case_rows,
                          'expected_counts': counts,
                          'expected_runtime_sha256': {k: v['sha256'] for k, v in runtime.items()}}
        write_json(self.selection_path, self.selection)
        case = {'schema_version': 1, 'synthetic_fault_injection': True, 'source': source,
                'train_per_group': 1, 'eval_per_group': 1, 'rows': case_rows}
        write_json(self.dataset / 'CASE.json', case)
        public = {'schema_version': 1, 'synthetic_fault_injection': True, 'source': source,
                  'inputs': {'selection_sha256': sha(self.selection_path)},
                  'case_sha256': sha(self.dataset / 'CASE.json'),
                  'runtime_sha256': self.selection['expected_runtime_sha256'],
                  'runtime_sizes': {k: v['size'] for k, v in runtime.items()},
                  'verification': {'counts': counts}}
        write_json(self.dataset / 'PUBLIC_PROVENANCE.json', public)
        self.config = {'num_rollout': 8, 'rollout_batch_size': 8, 'n_samples_per_prompt': 2,
                       'global_batch_size': 16, 'seed': 20261031, 'synthetic_fixture': True}
        write_json(self.prefix / 'run/effective_config.json', self.config)
        self.state = {'cursor': 32, 'sample_index': 64, 'group_index': 32, 'step': 3, 'arm': 'faulty'}
        write_json(self.prefix / 'run/actor/rollout/data_state_3.json', self.state)
        self.checkpoint_evidence = {}
        for role in ('actor', 'critic'):
            base = self.prefix / 'run' / role
            base.mkdir(parents=True, exist_ok=True)
            (base / 'latest_checkpointed_iteration.txt').write_text('3\n')
            selected = base / 'iter_0000003'
            for name in ('.metadata', 'common.pt', 'metadata.json', '__0_0.distcp',
                         '__0_1.distcp', 'extra-state.bin', 'nested/additional.raw'):
                target = selected / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f'SYNTHETIC schema-only; NOT loadable: {role}/{name}\n'.encode())
            self.checkpoint_evidence[role] = {'iteration': '3', 'selected': selected.name,
                'referenced_files': inventory(selected), 'restore_tested': False,
                'state_keys': ['model.synthetic', 'optimizer.synthetic']}
        write_json(self.prefix / 'provenance/output_checkpoints.json', self.checkpoint_evidence)
        events = []
        for step in range(4):
            events.extend([{'kind': 'rollout_begin', 'step': step, 'cursor': step * 8, 'synthetic': True},
                           {'kind': 'rollout_end', 'step': step, 'synthetic': True}])
        write_lines(self.prefix / 'run/events.jsonl', events)
        for role in ('actor', 'critic'):
            write_lines(self.prefix / f'run/optimizer_evidence/{role}-rank0-pid123.jsonl', [
                {'event': 'optimizer_step', 'role': role, 'rollout_id': step,
                 'update_successful': True, 'synthetic': True} for step in range(4)])
        self.inputs = {'synthetic_fault_injection': True, 'arm': 'faulty', 'steps': 8, 'relocate_at': 4,
                       'model_revision': 'b' * 40, 'source_revision': 'c' * 40,
                       'model_files': {'model.bin': 'd' * 64}, 'native_checkpoint_files': {},
                       'manifest_sha256': sha(self.dataset / 'manifest.json'),
                       'eval_sha256': sha(self.dataset / 'eval.jsonl'),
                       'case_sha256': sha(self.dataset / 'CASE.json')}
        write_json(self.prefix / 'INPUTS.json', self.inputs)
        # Native conversion, unrelated dumps, and prepared models are deliberately excluded.
        write_json(self.prefix / 'native/conversion_receipt.json', {'synthetic': True})
        (self.prefix / 'native/model.bin').write_bytes(b'SYNTHETIC conversion, excluded')
        (self.prefix / 'run/rollout_3.pt').write_bytes(b'SYNTHETIC debug dump, excluded')

    def module(self):
        self.assertTrue(SCRIPT.is_file(), 'scene assembly implementation is missing')
        spec = importlib.util.spec_from_file_location('incident_scene', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def create(self, **kwargs):
        return self.module().create_scene(self.prefix, self.scene,
                                          selection_path=self.selection_path, **kwargs)

    def test_create_moves_only_copied_hard_files_and_preserves_manifest(self):
        before = inventory(self.prefix)
        receipt = self.create()
        self.assertEqual(inventory(self.prefix), before)
        self.assertEqual(receipt, json.loads((self.scene / 'SCENE.json').read_text()))
        self.assertEqual((self.scene / 'dataset/manifest.json').read_bytes(),
                         (self.dataset / 'manifest.json').read_bytes())
        self.assertFalse((self.scene / 'dataset/live/hard').exists())
        for row in self.rows:
            name = row['path'].replace('live/hard/', 'archive/hard/')
            copied = self.scene / 'dataset' / name
            self.assertEqual(sha(copied), row['sha256'])
            self.assertFalse(copied.samefile(self.dataset / row['path']))
        self.assertEqual(self.module().verify_scene(self.scene), receipt)

    def test_checkpoint_inventory_includes_all_bytes_and_cursor_without_hardlinks(self):
        receipt = self.create()
        for role in ('actor', 'critic'):
            original = self.prefix / 'run' / role
            copied = self.scene / 'checkpoints' / role
            self.assertEqual(inventory(original), inventory(copied))
            for name in inventory(original):
                self.assertIn(f'checkpoints/{role}/{name}', receipt['files'])
                self.assertFalse((original / name).samefile(copied / name))
        self.assertFalse((self.scene / 'native').exists())
        self.assertFalse((self.scene / 'run/rollout_3.pt').exists())
        self.assertFalse(receipt['restore_tested'])

    def test_receipt_records_resume_parameters_original_hashes_and_sorted_inventory(self):
        receipt = self.create()
        for key, value in {'schema_version': 1, 'synthetic_fault_injection': True,
                           'checkpoint_iteration': 3, 'resume_start': 4, 'final_step': 7,
                           'sample_cursor': 32, 'sample_index': 64, 'global_batch_size': 16}.items():
            self.assertEqual(receipt[key], value, key)
        self.assertEqual(receipt['source']['inputs'], self.inputs)
        self.assertEqual(receipt['source']['parameters'], self.config)
        self.assertEqual(receipt['source']['files']['dataset/manifest.json']['sha256'],
                         self.inputs['manifest_sha256'])
        self.assertEqual(list(receipt['files']), sorted(receipt['files']))
        actual = inventory(self.scene)
        del actual['SCENE.json']
        self.assertEqual(receipt['files'], actual)

    def test_catalog_uses_arbitrary_relative_destination_and_all_original_ids(self):
        receipt = self.create(move_subdir='cold/tier2/selected')
        catalog = json.loads((self.scene / 'dataset/catalog.json').read_text())
        self.assertEqual({row['id'] for row in catalog['rows']}, {row['id'] for row in self.rows})
        for row in catalog['rows']:
            expected = 'cold/tier2/selected/' if row['difficulty'] == 'hard' else 'live/easy/'
            self.assertTrue(row['path'].startswith(expected))
            self.assertEqual(sha(self.scene / 'dataset' / row['path']), row['sha256'])
        self.assertEqual(self.module().verify_scene(self.scene), receipt)

    def test_parameters_come_from_actual_metadata_with_tiny_dataset(self):
        self.config.update(rollout_batch_size=2, global_batch_size=4)
        self.state.update(cursor=8, group_index=8, sample_index=16)
        write_json(self.prefix / 'run/effective_config.json', self.config)
        write_json(self.prefix / 'run/actor/rollout/data_state_3.json', self.state)
        events = [{'kind': kind, 'step': step, 'cursor': step * 2, 'synthetic': True}
                  for step in range(4) for kind in ('rollout_begin', 'rollout_end')]
        write_lines(self.prefix / 'run/events.jsonl', events)
        receipt = self.create()
        self.assertEqual((receipt['sample_cursor'], receipt['sample_index'], receipt['global_batch_size']),
                         (8, 16, 4))

    def test_default_public_selection_rejects_synthetic_source_ids(self):
        with self.assertRaises(ValueError):
            self.module().create_scene(self.prefix, self.scene)
        self.assertFalse(self.scene.exists())

    def test_wrong_selected_source_id_fails_before_copy(self):
        self.selection['rows'][0]['uuid'] = 'WRONG_SOURCE'
        write_json(self.selection_path, self.selection)
        public = json.loads((self.dataset / 'PUBLIC_PROVENANCE.json').read_text())
        public['inputs']['selection_sha256'] = sha(self.selection_path)
        write_json(self.dataset / 'PUBLIC_PROVENANCE.json', public)
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(self.scene.exists())

    def test_missing_referenced_checkpoint_file_fails_before_copy(self):
        (self.prefix / 'run/critic/iter_0000003/__0_1.distcp').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.create()
        self.assertFalse(self.scene.exists())

    def test_wrong_tracker_or_cursor_fails_before_copy(self):
        for kind in ('tracker', 'cursor'):
            with self.subTest(kind=kind):
                tracker = self.prefix / 'run/critic/latest_checkpointed_iteration.txt'
                tracker.write_text('4\n' if kind == 'tracker' else '3\n')
                if kind == 'cursor':
                    self.state['sample_index'] = 0
                    write_json(self.prefix / 'run/actor/rollout/data_state_3.json', self.state)
                with self.assertRaises(ValueError):
                    self.create()
                self.assertFalse(self.scene.exists())

    def test_missing_prefix_step_evidence_is_rejected(self):
        write_lines(self.prefix / 'run/events.jsonl', [{'kind': 'rollout_end', 'step': 3}])
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(self.scene.exists())

    def test_nonempty_destination_is_not_overwritten(self):
        self.scene.mkdir()
        (self.scene / 'keep').write_text('user data')
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual((self.scene / 'keep').read_text(), 'user data')

    def test_exports_dataset_and_role_relative_checkpoint_inventories(self):
        receipt = self.create()
        self.assertEqual(receipt['dataset']['path'], 'dataset')
        self.assertEqual(receipt['dataset']['files'], inventory(self.scene / 'dataset'))
        self.assertEqual(receipt['dataset']['train_ids'], [row['id'] for row in self.rows])
        for role in ('actor', 'critic'):
            checkpoint = receipt['checkpoints'][role]
            self.assertEqual(checkpoint['path'], f'checkpoints/{role}')
            self.assertEqual(checkpoint['iteration'], 3)
            self.assertEqual(checkpoint['selected'], 'iter_0000003')
            self.assertEqual(checkpoint['files'], inventory(self.scene / 'checkpoints' / role))
        self.assertEqual(receipt['datasource_state']['path'], 'checkpoints/actor/rollout/data_state_3.json')
        self.assertEqual(receipt['datasource_state']['sha256'], sha(self.prefix / 'run/actor/rollout/data_state_3.json'))

    def test_healthy_prefix_cursor_arm_is_preserved_exactly(self):
        self.inputs['arm'] = self.state['arm'] = 'healthy'
        write_json(self.prefix / 'INPUTS.json', self.inputs)
        original = self.prefix / 'run/actor/rollout/data_state_3.json'
        write_json(original, self.state)
        self.create()
        self.assertEqual((self.scene / 'checkpoints/actor/rollout/data_state_3.json').read_bytes(),
                         original.read_bytes())
        self.module().verify_scene(self.scene)

    def test_copied_scene_is_independent_and_verifies_without_prefix(self):
        receipt = self.create()
        destination = self.root / 'copy'
        copied = self.module().copy_scene(self.scene, destination)
        self.assertEqual(copied, receipt)
        self.assertEqual(inventory(destination), inventory(self.scene))
        for name in inventory(destination):
            self.assertFalse((destination / name).samefile(self.scene / name))
        shutil.rmtree(self.prefix)
        self.selection_path.unlink()
        self.assertEqual(self.module().verify_scene(destination), receipt)

    def test_materialize_uses_incident_subdir_and_preserves_workspace_files(self):
        receipt = self.create()
        workspace = self.root / 'workspace'
        workspace.mkdir()
        (workspace / 'loader.py').write_text('# user resolver\n')
        self.assertEqual(self.module().materialize_scene(self.scene, workspace), receipt)
        self.assertEqual(self.module().verify_scene(workspace / 'incident'), receipt)
        self.assertEqual((workspace / 'loader.py').read_text(), '# user resolver\n')
        with self.assertRaises(FileExistsError):
            self.module().materialize_scene(self.scene, workspace)

    def test_copy_rejects_nonempty_destination_and_invalid_source(self):
        self.create()
        destination = self.root / 'copy'
        destination.mkdir()
        (destination / 'keep').write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.module().copy_scene(self.scene, destination)
        (self.scene / 'checkpoints/critic/iter_0000003/extra-state.bin').write_bytes(b'changed')
        empty = self.root / 'invalid-copy'
        with self.assertRaises(ValueError):
            self.module().copy_scene(self.scene, empty)
        self.assertFalse(empty.exists())

    def test_tampering_missing_files_and_extra_files_are_detected(self):
        self.create()
        target = self.scene / 'checkpoints/actor/iter_0000003/common.pt'
        original = target.read_bytes()
        for operation in ('tamper', 'missing', 'extra'):
            with self.subTest(operation=operation):
                if operation == 'tamper':
                    target.write_bytes(b'changed checkpoint')
                elif operation == 'missing':
                    target.unlink()
                else:
                    (self.scene / 'unexpected.txt').write_text('unrecorded')
                with self.assertRaises(ValueError):
                    self.module().verify_scene(self.scene)
                target.write_bytes(original)
        (self.scene / 'unexpected.txt').unlink()
        self.module().verify_scene(self.scene)

    def test_incomplete_catalog_rejected_even_if_its_inventory_is_refreshed(self):
        receipt = self.create()
        path = self.scene / 'dataset/catalog.json'
        catalog = json.loads(path.read_text())
        catalog['rows'].pop()
        write_json(path, catalog)
        info = {'size': path.stat().st_size, 'sha256': sha(path)}
        receipt['files']['dataset/catalog.json'] = info
        receipt['dataset']['files']['catalog.json'] = info
        write_json(self.scene / 'SCENE.json', receipt)
        with self.assertRaisesRegex(ValueError, '[Cc]atalog'):
            self.module().verify_scene(self.scene)

    def test_receipt_cursor_and_tracker_semantics_are_checked(self):
        receipt = self.create()
        receipt['sample_cursor'] = 0
        write_json(self.scene / 'SCENE.json', receipt)
        with self.assertRaises(ValueError):
            self.module().verify_scene(self.scene)

    def test_archive_is_complete_verified_and_never_contains_links(self):
        receipt = self.create()
        for suffix in ('.tar', '.tar.gz'):
            with self.subTest(suffix=suffix):
                archive = self.root / ('scene' + suffix)
                packed = self.module().pack_scene(self.scene, archive)
                self.assertEqual(packed['sha256'], sha(archive))
                self.assertEqual(packed['file_count'], len(receipt['files']) + 1)
                with tarfile.open(archive, 'r:*') as tar:
                    self.assertEqual(set(tar.getnames()), set(receipt['files']) | {'SCENE.json'})
                    for member in tar.getmembers():
                        self.assertTrue(member.isfile())
                        self.assertEqual(tar.extractfile(member).read(), (self.scene / member.name).read_bytes())
                with self.assertRaises(FileExistsError):
                    self.module().pack_scene(self.scene, archive)

    def test_cli_create_verify_copy_and_pack_emit_json(self):
        self.module()
        commands = [(['create', '--prefix', str(self.prefix), '--destination', str(self.scene),
                      '--selection', str(self.selection_path), '--move-subdir', 'cold/hard'], self.scene),
                    (['verify', str(self.scene)], self.scene),
                    (['copy', str(self.scene), str(self.root / 'cli-copy')], self.root / 'cli-copy'),
                    (['pack', str(self.scene), str(self.root / 'cli-scene.tar')], None)]
        for args, destination in commands:
            with self.subTest(command=args[0]):
                result = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), *args],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertTrue(summary)
                if destination:
                    self.module().verify_scene(destination)

    def test_assembly_is_deterministic_for_identical_prefix_bytes(self):
        first = self.create()
        second = self.module().create_scene(self.prefix, self.root / 'second', selection_path=self.selection_path)
        self.assertEqual(second, first)

    def test_no_checkpoint_inspector_is_required_when_no_external_inventory_exists(self):
        (self.prefix / 'provenance/output_checkpoints.json').unlink()
        receipt = self.create()
        self.assertFalse(receipt['restore_tested'])
        self.assertEqual(self.module().verify_scene(self.scene), receipt)


if __name__ == '__main__':
    unittest.main()
