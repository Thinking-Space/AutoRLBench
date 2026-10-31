"""Portable Python 3.12 offline tests; mutations are isolated synthetic fixtures.

Default evidence comes from the checksum-verified bundled recorded-audit archive.
Set RELOCATION_RECORDED_ROOT explicitly to use another recorded-home directory
with job-ID or arm subdirectories. Frozen evidence is never modified.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
RECORDED_ARMS = {'1428042': 'healthy', '1428043': 'faulty', '1428044': 'repaired'}


def load_grader():
    path = HERE / 'grade_submission.py'
    assert path.is_file(), 'The solution grader has not been implemented'
    spec = importlib.util.spec_from_file_location('relocation_solution_grader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def recorded_homes(add_cleanup, override=None):
    """Load real recorded fixtures; missing or changed evidence fails, never skips."""
    if override:
        root = Path(override).expanduser()
    else:
        relative = Path('proposals/justrl2_data_relocation/evidence/recorded-audit.tar.gz')
        archive = next((parent / relative for parent in HERE.parents if (parent / relative).is_file()), None)
        if archive is None:
            raise FileNotFoundError('Bundled recorded-audit.tar.gz is missing; restore proposal evidence '
                                    'or explicitly set RELOCATION_RECORDED_ROOT')
        digest = read_json(archive.parent / 'sha256.json')[archive.name]

        def verify_archive():
            if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise AssertionError('Bundled recorded archive checksum mismatch')

        verify_archive()
        temporary = tempfile.TemporaryDirectory(prefix='frozen-recorded-audit-')
        add_cleanup(temporary.cleanup)
        with tarfile.open(archive, 'r:gz') as bundle:
            bundle.extractall(temporary.name, filter='data')
        root = Path(temporary.name) / 'recorded'
        manifest = root / 'sha256.json'
        pins = read_json(manifest)
        pins['sha256.json'] = hashlib.sha256(manifest.read_bytes()).hexdigest()

        def verify_frozen_files():
            verify_archive()
            files = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
            if files != set(pins):
                raise AssertionError('Frozen recorded evidence file set changed')
            for name, expected in pins.items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise AssertionError(f'Frozen recorded evidence checksum mismatch: {name}')

        verify_frozen_files()
        # unittest cleanups are LIFO: verify before removing the temporary tree.
        add_cleanup(verify_frozen_files)
    homes = {}
    for job, arm in RECORDED_ARMS.items():
        home = root / job if (root / job).is_dir() else root / arm
        inputs = read_json(home / 'INPUTS.json')
        if inputs.get('job_id') != job or inputs.get('arm') != arm:
            raise AssertionError(f'Recorded home identity mismatch for {job}/{arm}')
        homes[job] = home
    return homes


class RecordedCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grader = load_grader()
        cls.homes = recorded_homes(cls.addClassCleanup, os.environ.get('RELOCATION_RECORDED_ROOT'))

    def test_actual_recorded_arms(self):
        for job, expected in [('1428042', 0), ('1428043', 0), ('1428044', 1)]:
            with self.subTest(job=job):
                self.assertEqual(self.grader.grade_solution(self.homes[job]), expected)

    def copy_home(self, job='1428044'):
        temporary = tempfile.TemporaryDirectory(prefix='synthetic-grader-test-')
        self.addCleanup(temporary.cleanup)
        home = Path(temporary.name) / 'home'
        home.mkdir()
        source = self.homes[job]
        for name in ('dataset', 'provenance', 'run'):
            shutil.copytree(source / name, home / name)
        shutil.copyfile(source / 'INPUTS.json', home / 'INPUTS.json')
        for name in ('RESULT.json', 'export/RESULT.json'):
            if (source / name).exists():
                (home / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, home / name)
        return home

    def events(self, home):
        return [json.loads(line) for line in (home / 'run/events.jsonl').read_text().splitlines()]

    def write_events(self, home, events):
        (home / 'run/events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))

    def relabel(self, home, arm):
        for name in ('INPUTS.json', 'run/TRAINING_RETURNED.json', 'run/actor/rollout/data_state_7.json'):
            data = read_json(home / name)
            data['arm'] = arm
            write_json(home / name, data)
        config = read_json(home / 'run/effective_config.json')
        config['train_env_vars']['RELOCATION_ARM'] = arm
        write_json(home / 'run/effective_config.json', config)
        events = self.events(home)
        for row in events:
            row['arm'] = arm
        self.write_events(home, events)

    def test_repaired_behavior_under_faulty_label_passes(self):
        home = self.copy_home()
        self.relabel(home, 'faulty')
        self.assertEqual(self.grader.grade_solution(home), 1)

    def test_archive_resolution_without_manifest_rebinding_passes(self):
        home = self.copy_home()
        self.relabel(home, 'faulty')
        initial = read_json(home / 'provenance/initial_manifest.json')
        for name in ('dataset/manifest.json', 'provenance/final_manifest.json'):
            write_json(home / name, initial)
        events = self.events(home)
        receipt = next(row['receipt'] for row in events if row['kind'] == 'relocation')
        receipt['repair'] = False
        receipt['manifest_after_sha256'] = receipt['manifest_before_sha256']
        for row in events:
            if row['kind'] == 'relocation':
                row['receipt'] = receipt
        self.write_events(home, events)
        for name in ('dataset/relocation.json', 'provenance/final_relocation.json'):
            write_json(home / name, {'schema_version': 1, 'pending': None, 'events': [receipt]})
        self.assertEqual(self.grader.grade_solution(home), 1)

    def test_event_arm_mismatch_fails(self):
        home = self.copy_home()
        events = self.events(home)
        events[0]['arm'] = 'faulty'
        self.write_events(home, events)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_source_label_mismatch_fails(self):
        home = self.copy_home()
        events = self.events(home)
        for row in events:
            if row['kind'] in ('terminal', 'admitted') and row['index'] == 0:
                row['label'] = 'synthetic wrong answer'
        self.write_events(home, events)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_dropped_hard_source_fails(self):
        home = self.copy_home()
        manifest = read_json(home / 'dataset/manifest.json')
        manifest['rows'] = [row for row in manifest['rows'] if row['difficulty'] != 'hard']
        for name in ('dataset/manifest.json', 'provenance/final_manifest.json'):
            write_json(home / name, manifest)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_suppressed_metrics_fail(self):
        for metric in ('data/abort_rate', 'data/terminal_count', 'data/attempted_hard'):
            with self.subTest(metric=metric):
                home = self.copy_home()
                events = self.events(home)
                next(row for row in events if row['kind'] == 'rollout_end' and row['step'] == 4)['metrics'].pop(metric)
                self.write_events(home, events)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_missing_evidence_fails(self):
        for relative in ('run/TRAINING_RETURNED.json', 'export/RESULT.json',
                         'run/eval_after_step_0000007.json', 'provenance/output_checkpoints.json',
                         'run/actor/rollout/data_state_7.json', 'dataset/relocation.json'):
            with self.subTest(relative=relative):
                home = self.copy_home()
                (home / relative).unlink()
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_root_failure_is_not_hidden_by_successful_export(self):
        home = self.copy_home()
        result = read_json(home / 'export/RESULT.json')
        result['returncode'] = 1
        write_json(home / 'RESULT.json', result)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_fixed_budgets_cannot_change(self):
        for key, value in [('rollout_batch_size', 4), ('rollout_max_response_len', 8192),
                           ('n_samples_per_eval_prompt', 2), ('lr', 1e-4), ('debug_disable_optimizer', True)]:
            with self.subTest(key=key):
                home = self.copy_home()
                config = read_json(home / 'run/effective_config.json')
                config[key] = value
                write_json(home / 'run/effective_config.json', config)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_removed_terminal_and_cursor_gap_fail(self):
        for mutation in ('terminal', 'cursor', 'step'):
            with self.subTest(mutation=mutation):
                home = self.copy_home()
                events = self.events(home)
                if mutation == 'terminal':
                    events.remove(next(row for row in events if row['kind'] == 'terminal'))
                elif mutation == 'cursor':
                    next(row for row in events if row['kind'] == 'rollout_begin' and row['step'] == 4)['cursor'] += 1
                else:
                    events = [row for row in events if row['step'] != 7]
                self.write_events(home, events)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_native_update_count_is_exact(self):
        for mutation in ('missing', 'extra'):
            with self.subTest(mutation=mutation):
                home = self.copy_home()
                path = next((home / 'run/optimizer_evidence').glob('actor-*.jsonl'))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mutation == 'missing':
                    rows.pop()
                else:
                    rows.append(dict(rows[0], step_id=999))
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_final_checkpoint_requires_model_optimizer_and_step_seven(self):
        for mutation in ('iteration', 'optimizer', 'model'):
            with self.subTest(mutation=mutation):
                home = self.copy_home()
                path = home / 'provenance/output_checkpoints.json'
                checkpoints = read_json(path)
                record = checkpoints['actor']
                if mutation == 'iteration':
                    record.update(iteration='3', selected='iter_0000003')
                elif mutation == 'optimizer':
                    record['state_keys'] = [key for key in record['state_keys'] if 'optim' not in key.lower()]
                else:
                    record['state_keys'] = ['optimizer.state']
                write_json(path, checkpoints)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_original_native_hash_change_fails(self):
        home = self.copy_home()
        path = home / 'provenance/original_native_unchanged.json'
        inventory = read_json(path)
        inventory['.metadata']['sha256'] = '0' * 64
        write_json(path, inventory)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_eval_label_and_prompt_are_bound_to_public_selection(self):
        for field in ('label', 'prompt'):
            with self.subTest(field=field):
                home = self.copy_home()
                for path in (home / 'run').glob('eval_*.json'):
                    output = read_json(path)
                    output['fixed']['samples'][0][field] = 'synthetic mismatch'
                    write_json(path, output)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_no_heldout_accuracy_threshold(self):
        home = self.copy_home()
        for path in (home / 'run').glob('eval_*.json'):
            output = read_json(path)
            output['fixed']['rewards'] = [0] * 16
            for sample in output['fixed']['samples']:
                sample['reward'] = 0
            write_json(path, output)
        self.assertEqual(self.grader.grade_solution(home), 1)

    def test_eval_dataset_budget_is_fixed(self):
        home = self.copy_home()
        path = home / 'run/effective_config.json'
        config = read_json(path)
        config['eval_datasets'][0] = config['eval_datasets'][0].replace('max_response_len=32768', 'max_response_len=8192')
        write_json(path, config)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_changed_archive_bytes_fail(self):
        home = self.copy_home()
        path = next((home / 'dataset/archive/hard').glob('*.json'))
        payload = read_json(path)
        payload['label'] = 'synthetic incorrect label'
        write_json(path, payload)
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_model_and_source_pins_are_fixed(self):
        for key in ('model_revision', 'source_revision', 'model_files'):
            with self.subTest(key=key):
                home = self.copy_home()
                path = home / 'INPUTS.json'
                inputs = read_json(path)
                if key == 'model_files':
                    inputs[key]['model.safetensors'] = '0' * 64
                else:
                    inputs[key] = '0' * 40
                write_json(path, inputs)
                self.assertEqual(self.grader.grade_solution(home), 0)

    def test_fresh_checkpoint_payload_contract(self):
        home = self.copy_home()
        self.relabel(home, 'faulty')
        with self.assertRaisesRegex(ValueError, 'payloads'):
            self.grader.inspect_solution(home, fresh=True)

        def synthetic_payloads(root, record, tag):
            # Explicitly synthetic bytes test file-integrity checks, not native
            # checkpoint semantics or actual training. Never touch raw homes.
            inventory = {}
            for name in record['referenced_files']:
                path = root / record['selected'] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                content = f'ISOLATED SYNTHETIC TEST PAYLOAD: {tag}/{name}\n'.encode()
                path.write_bytes(content)
                inventory[name] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            record['referenced_files'] = inventory
            return inventory

        checkpoints = read_json(home / 'provenance/output_checkpoints.json')
        for role, record in checkpoints.items():
            synthetic_payloads(home / 'run' / role, record, role)
        write_json(home / 'provenance/output_checkpoints.json', checkpoints)
        conversion = read_json(home / 'provenance/conversion_receipt.json')
        inventory = synthetic_payloads(home / 'native', conversion['native_checkpoint'], 'original')
        write_json(home / 'provenance/conversion_receipt.json', conversion)
        inputs = read_json(home / 'INPUTS.json')
        inputs['native_checkpoint_files'] = inventory
        write_json(home / 'INPUTS.json', inputs)
        write_json(home / 'provenance/original_native_unchanged.json', inventory)
        full = {'release/' + name: row for name, row in inventory.items()}
        for name in ('native_all_files_before.json', 'native_all_files_after.json'):
            write_json(home / 'provenance' / name, full)
        self.assertEqual(self.grader.inspect_solution(home, fresh=True)['structural_coverage'], 1)

        # The default CLI must read this exact fresh log directory, with no
        # offline flag or dependency on files outside the run home.
        results = home.parent / 'results'
        destination = results / 'experiment_logs/run-home'
        destination.parent.mkdir(parents=True)
        home.rename(destination)
        completed = subprocess.run(
            [sys.executable, '-B', str(HERE / 'grade_submission.py'), '--results', str(results)],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(read_json(results / 'metrics.json')['offline_calibration'])
        native_file = destination / 'native/release/__0_0.distcp'
        native_file.write_bytes(b'ISOLATED SYNTHETIC TEST: altered original native bytes')
        with self.assertRaisesRegex(ValueError, 'payload hash/size mismatch'):
            self.grader.inspect_solution(destination, fresh=True)

    def test_malformed_input_fails_closed(self):
        home = self.copy_home()
        (home / 'INPUTS.json').write_text('{incomplete')
        self.assertEqual(self.grader.grade_solution(home), 0)

    def test_cli_offline_outputs_and_fresh_log_scope(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-grader-output-') as temporary:
            results = Path(temporary)
            command = [sys.executable, '-B', str(HERE / 'grade_submission.py')]
            env = dict(os.environ, AUTORLBENCH_RESULTS=str(results), PYTHONDONTWRITEBYTECODE='1')
            completed = subprocess.run(command + ['--home', str(self.homes['1428044'])], env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in ('result.json', 'metrics.json', 'experiment_metrics.json'):
                self.assertTrue((results / name).is_file())
            self.assertEqual(read_json(results / 'result.json')['score'], 1)
            self.assertEqual(json.loads(completed.stdout)['score'], 1)
            # Previous outputs and recorded evidence outside fresh logs cannot pass.
            completed = subprocess.run(command + ['--results', str(results)], env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertEqual(read_json(results / 'result.json')['score'], 0)


class MissingInput(unittest.TestCase):
    def test_nonexistent_home_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(load_grader().grade_solution(Path(temporary) / 'missing'), 0)


class FixturePortability(unittest.TestCase):
    def test_bundled_frozen_runtime_is_unchanged(self):
        homes = recorded_homes(self.addCleanup, os.environ.get('RELOCATION_RECORDED_ROOT'))
        reference = read_json(HERE.parent / 'resources/reference_selection.json')
        pins = reference['expected_runtime_sha256']
        self.assertEqual(len(pins), 66)
        for job, home in homes.items():
            self.assertEqual(read_json(home / 'INPUTS.json')['job_id'], job)
            for relative, digest in pins.items():
                with self.subTest(job=job, relative=relative):
                    path = home / 'dataset' / relative
                    if relative == 'manifest.json':
                        path = home / 'provenance/initial_manifest.json'
                    elif relative.startswith('live/hard/') and not path.exists():
                        path = home / 'dataset/archive/hard' / Path(relative).name
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        # Mutations must affect disposable copies only, never frozen fixtures.
        original = next((homes['1428044'] / 'dataset/archive/hard').glob('*.json'))
        frozen_bytes = original.read_bytes()
        with tempfile.TemporaryDirectory(prefix='synthetic-portability-test-') as temporary:
            copied = Path(temporary) / original.name
            shutil.copyfile(original, copied)
            copied.write_text('ISOLATED SYNTHETIC MUTATION\n')
            self.assertEqual(original.read_bytes(), frozen_bytes)


if __name__ == '__main__':
    unittest.main()
