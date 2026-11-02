"""Task packaging contracts, using only pinned public files and local fixtures."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/data/justrl2_data_relocation'
SOURCE = ROOT / 'proposals/justrl2_data_relocation/incident'
ENGINE = TASK / 'environment/incident_engine'
VERIFIER = TASK / 'evaluations/solution/verifier'
AUDIT = VERIFIER / 'resources/incident_audit'
SCENE_SHA = 'b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac'
VERIFIER_SHA = 'db21b6dd204f55efb4e4ce06aa65dfb0a7dd76e9ac35caa87fc31223cbfe4286'
RUNTIME_FILES = {
    'accounting.py', 'loader_contract.py', 'materialize.py', 'resolve_worker.py',
    'restore_evidence.py', 'rollout.py', 'run.py', 'run_native.py', 'scene.py',
    'stage.py', 'stage_runtime.py', 'store.py', 'train.sh', 'verify_resume.py',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}


def synthetic_fixture(root):
    """Reuse labeled schema fixtures only; never read an actual scene/checkpoint."""
    spec = importlib.util.spec_from_file_location('_packaging_fixture', SOURCE / 'test_verify_resume.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Fixture(root.resolve())


def clean_env(**values):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('AUTORLBENCH_', 'RELOCATION_', 'BENCH_RESTORE_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME')}
    return dict(env, PYTHONDONTWRITEBYTECODE='1', **{k: str(v) for k, v in values.items()})


def cached_prepare_fixture(root, *, cache=True):
    """Temporary synthetic scene and prepared assets; no downloads or model work."""
    fixture = synthetic_fixture(root / 'fixture')
    environment, prepared, workspace = root / 'environment', root / 'prepared', root / 'workspace'
    environment.mkdir(); prepared.mkdir(); workspace.mkdir()
    shutil.copytree(ENGINE, environment / 'incident_engine')
    for name in ('setup_incident.py', 'prepare.sh'):
        shutil.copyfile(TASK / 'environment' / name, environment / name)
    pin = dict(scene_sha256=sha(fixture.scene / 'SCENE.json'), synthetic_fixture=True)
    (environment / 'snapshot.json').write_text(json.dumps(pin))
    datasets = root / 'datasets cache'
    datasets.mkdir()
    cached = datasets / 'incidents' / pin['scene_sha256']
    if cache:
        shutil.copytree(fixture.scene, cached)
    env = clean_env(AUTORLBENCH_ENVIRONMENT=environment, AUTORLBENCH_PREPARED=prepared,
                    AUTORLBENCH_WORKSPACE=workspace, AUTORLBENCH_DATASETS=datasets)
    return fixture, environment, prepared, workspace, cached, env


class TaskPackaging(unittest.TestCase):
    def test_runtime_allowlist_matches_frozen_source(self):
        expected = RUNTIME_FILES | {'snapshot.json'} | {
            'runtime_support/' + p.name for p in (SOURCE / 'runtime_support').iterdir() if p.is_file()}
        self.assertEqual(set(files(ENGINE)), expected)
        for name in expected - {'snapshot.json'}:
            with self.subTest(name=name):
                self.assertEqual((ENGINE / name).read_bytes(), (SOURCE / name).read_bytes())
        self.assertEqual(sha(ENGINE / 'verify_resume.py'), VERIFIER_SHA)

    def test_hidden_auditor_is_complete_and_frozen(self):
        self.assertEqual(set(files(AUDIT)), {'verify_resume.py', 'accounting.py', 'scene.py'})
        for name in files(AUDIT):
            self.assertEqual((AUDIT / name).read_bytes(), (SOURCE / name).read_bytes())
        self.assertEqual(sha(AUDIT / 'verify_resume.py'), VERIFIER_SHA)

    def test_scene_identity_is_unchanged(self):
        for path in (TASK / 'environment/snapshot.json', ENGINE / 'snapshot.json',
                     VERIFIER / 'resources/snapshot.json'):
            pin = json.loads(path.read_text())
            self.assertEqual(pin['scene_sha256'], SCENE_SHA)
            self.assertEqual((pin['checkpoint_iteration'], pin['resume_start'], pin['final_step']), (3, 4, 7))

    def test_reference_and_authoring_tools_are_not_installed(self):
        for directory in (ENGINE, TASK / 'workspace'):
            for name in files(directory):
                self.assertNotIn(Path(name).name, {'reference_loader.py', 'build.py', 'prefix.py'})
                self.assertFalse(Path(name).name.startswith('test_'), name)
        oracle = TASK / 'evaluations/solution/oracle/reference_loader.py'
        self.assertEqual(oracle.read_bytes(), (SOURCE / 'reference_loader.py').read_bytes())
        self.assertEqual((TASK / 'workspace/loader.py').read_bytes(), (SOURCE / 'loader.py').read_bytes())

    def test_manifest_describes_every_current_task_file(self):
        manifest = json.loads((TASK / 'provenance/source_manifest.json').read_text())
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(manifest['scene_sha256'], SCENE_SHA)
        rows = manifest['files']
        by_destination = {r['destination']: r for r in rows}
        self.assertEqual(len(rows), len(by_destination))
        self.assertEqual(set(by_destination), set(files(TASK)) - {'provenance/source_manifest.json'})
        for name, row in by_destination.items():
            with self.subTest(name=name):
                self.assertEqual(sha(TASK / name), row['after_sha256'])
                if row['source'] is not None:
                    self.assertEqual(sha(ROOT / row['source']), row['before_sha256'])

    def test_historical_validation_cannot_claim_incident_acceptance(self):
        current = json.loads((TASK / 'provenance/validation.json').read_text())
        self.assertEqual(current['phase'], 'incident_checkpoint_continuation')
        verified = current['incident_gpu_acceptance_verified']
        self.assertIs(type(verified), bool)
        if verified:
            run = current['packaged_task_reference_run']
            self.assertEqual(run['job_id'], 1430440)
            self.assertEqual(run['status'], 'Succeeded')
            self.assertIs(type(run['score']), int)
            self.assertEqual(run['score'], 1)
            receipt = run['receipt']
            self.assertEqual(receipt['path'], 'provenance/receipts/packaged-reference-grade.json')
            self.assertRegex(receipt['sha256'], r'^[0-9a-f]{64}$')
            receipt_path = TASK / receipt['path']
            self.assertEqual(sha(receipt_path), receipt['sha256'])
            grade = json.loads(receipt_path.read_text())
            self.assertIs(grade['passed'], True)
            self.assertIs(type(grade['score']), int)
            self.assertEqual(grade['score'], 1)
            proof = grade['metrics']
            self.assertEqual(proof['schema_version'], 3)
            self.assertIs(proof['score'], True)
            self.assertIs(proof['fresh'], True)
            self.assertEqual(proof['errors'], [])
            self.assertEqual(proof['scene_sha256'], SCENE_SHA)
            self.assertEqual(proof['scene_sha256'], current['scene_sha256'])
            self.assertIs(proof['original_scene_unchanged'], True)
            self.assertEqual(proof['admitted_count'], 64)
            self.assertEqual(set(proof['checkpoint_payloads_rehashed']), {'actor', 'critic'})
            self.assertIs(proof['model_assets']['relocated'], True)
            self.assertEqual(proof['model_assets']['files_rehashed'], 7)
            self.assertEqual(set(proof['optimizer']), {'actor', 'critic'})
            for role in ('actor', 'critic'):
                with self.subTest(role=role):
                    self.assertIs(proof['checkpoint_payloads_rehashed'][role], True)
                    optimizer = proof['optimizer'][role]
                    self.assertEqual(optimizer['steps'], [4, 5, 6, 7])
                    self.assertEqual(optimizer['restored_optimizer_step'], 4)
                    self.assertEqual(optimizer['saved_optimizer_step'], 8)
                    self.assertIs(optimizer['payload_exact'], True)
            self.assertEqual(proof['evaluation']['step'], 7)
            self.assertEqual(proof['evaluation']['count'], 16)
        else:
            self.assertEqual(current['incident_gpu_acceptance_status'], 'pending_fresh_packaged_runtime_audit')
        self.assertIs(current['agent_evaluation_executed'], False)
        history = TASK / current['historical_validation']['path']
        self.assertEqual(sha(history), current['historical_validation']['sha256'])
        self.assertEqual(json.loads(history.read_text())['fresh_grader']['job_id'], 1428910)
        oracle = TASK / 'evaluations/solution/oracle'
        self.assertFalse((oracle / 'reference_fresh_grade.json').exists())
        self.assertTrue((oracle / 'history/reference_fresh_grade.json').is_file())

    def test_prepare_materializes_the_scene_without_starting_training(self):
        script=(TASK/'environment/prepare.sh').read_text()
        self.assertIn('setup_incident.py',script)
        self.assertNotIn('run_case.py',script)
        self.assertTrue((TASK/'environment/incident_engine/run.py').is_file())

    def test_only_loader_is_the_submission_surface(self):
        self.assertTrue((TASK/'workspace/loader.py').is_file())
        self.assertFalse((TASK/'workspace/runtime').exists())
        self.assertFalse((TASK/'workspace/verify.py').exists())
        import json
        spec=json.loads((TASK/'evaluations/solution/verifier/resources/run_spec.json').read_text())
        self.assertIn('$AUTORLBENCH_PREPARED/incident_engine/run.py',spec['command'])
        self.assertIn('--loader "$AUTORLBENCH_WORKSPACE/loader.py"',spec['command'])
        self.assertNotIn('bash run.sh',spec['command'])

    def test_canonical_scaffolds_and_image_tooling_are_preserved(self):
        canonical = ROOT / 'tasks/reward/slime_retool_length_reward_bonus/evaluations'
        for name in ['verifier/test.sh', 'verifier/judge.py', 'verifier/judge_prompt.md']:
            self.assertEqual((TASK / 'evaluations/issue' / name).read_bytes(), (canonical / 'issue' / name).read_bytes())
        self.assertEqual((TASK / 'evaluations/solution/verifier/test.sh').read_bytes(),
                         (canonical / 'solution/verifier/test.sh').read_bytes())
        marker = '# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----'
        actual = (ROOT / 'docker/images/justrl2-data-relocation.Dockerfile').read_text().split(marker)[1]
        expected = (ROOT / 'docker/images/verl.Dockerfile').read_text().split(marker)[1]
        self.assertEqual(actual, expected)

    def test_runtime_entrypoints_import_without_repository_context(self):
        with tempfile.TemporaryDirectory(prefix='incident-cli-check-') as temporary:
            for directory, name in ((ENGINE, 'run.py'), (ENGINE, 'materialize.py'), (AUDIT, 'verify_resume.py')):
                command = [sys.executable, '-B', str(directory / name), '--help']
                if directory == AUDIT:
                    command.insert(1, '-I')
                result = subprocess.run(command, cwd=temporary, env=clean_env(), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_setup_installs_only_a_pinned_synthetic_scene_without_training(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-incident-install-') as temporary:
            root = Path(temporary)
            fixture = synthetic_fixture(root / 'fixture')
            environment, prepared, workspace = root / 'environment', root / 'prepared', root / 'workspace'
            environment.mkdir(); prepared.mkdir(); workspace.mkdir()
            shutil.copytree(ENGINE, environment / 'incident_engine')
            shutil.copyfile(TASK / 'environment/setup_incident.py', environment / 'setup_incident.py')
            pin = dict(scene_sha256=sha(fixture.scene / 'SCENE.json'), synthetic_fixture=True)
            (environment / 'snapshot.json').write_text(json.dumps(pin))
            before = {name: sha(path) for name, path in files(fixture.scene).items()}
            env = clean_env(AUTORLBENCH_ENVIRONMENT=environment, AUTORLBENCH_PREPARED=prepared,
                            AUTORLBENCH_WORKSPACE=workspace, RELOCATION_SCENE_SOURCE=fixture.scene)
            for _ in range(2):
                process = subprocess.run([sys.executable, '-B', str(environment / 'setup_incident.py')],
                    cwd=workspace, env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual((workspace / 'incident').resolve(), (prepared / 'incident').resolve())
                self.assertEqual(sha(prepared / 'incident/SCENE.json'), pin['scene_sha256'])
            self.assertEqual(before, {name: sha(path) for name, path in files(fixture.scene).items()})

    def test_prepare_sh_installs_from_harness_datasets_cache_without_custom_env(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-datasets-prepare-') as temporary:
            root = Path(temporary)
            fixture, environment, prepared, workspace, cached, env = cached_prepare_fixture(root)
            # Isolate the unrelated public asset download/conversion stage. The
            # actual prepare.sh -> setup -> materialize path still runs in full.
            (prepared / 'assets').mkdir()
            (prepared / 'assets/prepared_receipt.json').write_text('{"synthetic_fixture": true}\n')
            commands = root / 'commands'
            commands.mkdir()
            for name in ('autorlbench_fetch_hf', 'autorlbench_fetch_url'):
                command = commands / name
                command.write_text('#!/bin/sh\n# SYNTHETIC asset-stage stub only\nexit 0\n')
                command.chmod(0o755)
            (commands / 'python3').symlink_to(sys.executable)
            env['PATH'] = str(commands) + os.pathsep + env.get('PATH', '')
            self.assertNotIn('RELOCATION_SCENE_SOURCE', env)
            before = {name: sha(path) for name, path in files(cached).items()}
            for _ in range(2):
                process = subprocess.run(['bash', str(environment / 'prepare.sh')], cwd=workspace,
                    env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(sha(prepared / 'incident/SCENE.json'), sha(fixture.scene / 'SCENE.json'))
                self.assertEqual((workspace / 'incident').resolve(), (prepared / 'incident').resolve())
            self.assertEqual(before, {name: sha(path) for name, path in files(cached).items()})
            self.assertEqual(before, {name: sha(path) for name, path in files(prepared / 'incident').items()})

    def test_explicit_scene_source_takes_priority_over_valid_datasets_cache(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-explicit-scene-') as temporary:
            root = Path(temporary)
            _, environment, prepared, workspace, _, env = cached_prepare_fixture(root)
            explicit = root / 'missing-explicit-source'
            env['RELOCATION_SCENE_SOURCE'] = str(explicit)
            process = subprocess.run([sys.executable, '-B', str(environment / 'setup_incident.py')],
                cwd=workspace, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(str(explicit), process.stderr)
            self.assertFalse((prepared / 'incident').exists())

    def test_missing_datasets_scene_reports_expected_cache_path(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-missing-cache-') as temporary:
            _, environment, prepared, workspace, cached, env = cached_prepare_fixture(Path(temporary), cache=False)
            process = subprocess.run([sys.executable, '-B', str(environment / 'setup_incident.py')],
                cwd=workspace, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(str(cached), process.stderr)
            self.assertIn('--datasets-dir', process.stderr)
            self.assertFalse((prepared / 'incident').exists())

    def test_datasets_cache_directory_name_does_not_replace_scene_pin_check(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-wrong-cache-pin-') as temporary:
            root = Path(temporary)
            fixture, environment, prepared, workspace, _, env = cached_prepare_fixture(root, cache=False)
            wrong_pin = '0' * 64
            (environment / 'snapshot.json').write_text(json.dumps({'scene_sha256': wrong_pin}))
            shutil.copytree(fixture.scene, Path(env['AUTORLBENCH_DATASETS']) / 'incidents' / wrong_pin)
            process = subprocess.run([sys.executable, '-B', str(environment / 'setup_incident.py')],
                cwd=workspace, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('Source scene differs from artifact pin', process.stderr)
            self.assertFalse((prepared / 'incident').exists())

    def test_hidden_grader_accepts_complete_synthetic_evidence_then_rejects_missing_payload(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-packaged-grader-') as temporary:
            root = Path(temporary)
            fixture = synthetic_fixture(root / 'fixture')
            verifier, prepared, results = root / 'verifier', root / 'prepared', root / 'results'
            shutil.copytree(VERIFIER, verifier)
            prepared.mkdir(); (results / 'experiment_logs').mkdir(parents=True)
            (prepared / 'incident').symlink_to(fixture.scene)
            (results / 'experiment_logs/run-home').symlink_to(fixture.home)
            pin = dict(scene_sha256=sha(fixture.scene / 'SCENE.json'), synthetic_fixture=True)
            (verifier / 'resources/snapshot.json').write_text(json.dumps(pin))
            (results / 'report.md').write_text('Verdict: unhealthy\n\nChecked the current resolver and repaired missing-path resolution.\n')
            before = {name: sha(path) for name, path in files(fixture.scene).items()}
            for accepted in (True, False):
                if not accepted:
                    (fixture.home / 'run/actor/iter_0000007/__0_0.distcp').unlink()
                process = subprocess.run([sys.executable, '-I', '-B', str(verifier / 'tests/grade_submission.py')],
                    env=clean_env(AUTORLBENCH_PREPARED=prepared, AUTORLBENCH_RESULTS=results),
                    cwd=root, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0 if accepted else 1, process.stdout + process.stderr)
                result = json.loads((results / 'result.json').read_text())
                self.assertIs(result['passed'], accepted)
                self.assertIs(type(result['score']), int)
                self.assertEqual(result['score'], int(accepted))
                self.assertIs(result['metrics']['score'], accepted)
                self.assertEqual(json.loads((results / 'metrics.json').read_text()), result)
                if accepted:
                    self.assertEqual(result['metrics']['checkpoint_payloads_rehashed'], {'actor': True, 'critic': True})
            self.assertEqual(before, {name: sha(path) for name, path in files(fixture.scene).items()})

    def test_missing_scene_still_writes_a_zero_score_envelope(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC-missing-scene-') as temporary:
            root = Path(temporary)
            result_dir = root / 'results'
            process = subprocess.run([sys.executable, '-I', '-B', str(VERIFIER / 'tests/grade_submission.py')],
                env=clean_env(AUTORLBENCH_PREPARED=root / 'absent', AUTORLBENCH_RESULTS=result_dir),
                cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1)
            self.assertTrue((result_dir / 'result.json').is_file(), process.stderr)
            result = json.loads((result_dir / 'result.json').read_text())
            self.assertIs(result['passed'], False)
            self.assertEqual(result['score'], 0)
            self.assertTrue(result['metrics']['errors'])


if __name__ == '__main__':
    unittest.main()
