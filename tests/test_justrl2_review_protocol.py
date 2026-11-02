"""Behavioral checks for diagnosis-first submissions and the healthy control."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.test_justrl2_data_relocation import clean_env, files, synthetic_fixture

ROOT = Path(__file__).resolve().parents[1]
FAULTY = ROOT / 'tasks/data/justrl2_data_relocation'
HEALTHY = ROOT / 'tasks/data/justrl2_data_review_control'


def load(path):
    spec = importlib.util.spec_from_file_location('_review_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewProtocol(unittest.TestCase):
    def test_both_cases_have_the_same_neutral_prompt_and_environment(self):
        self.assertTrue(HEALTHY.is_dir())
        for level in ('issue', 'solution'):
            original = (FAULTY / f'evaluations/{level}/instruction.md').read_text()
            self.assertEqual(original, (HEALTHY / f'evaluations/{level}/instruction.md').read_text())
            self.assertIn('healthy', original)
            self.assertIn('unhealthy', original)
        for name in ('WORKSPACE.md', 'logs/recorded/README.md'):
            self.assertEqual((FAULTY / 'workspace' / name).read_bytes(), (HEALTHY / 'workspace' / name).read_bytes())
        visible_a, visible_b = files(FAULTY / 'environment'), files(HEALTHY / 'environment')
        self.assertEqual(set(visible_a), set(visible_b))
        self.assertTrue(all(visible_a[k].read_bytes() == visible_b[k].read_bytes() for k in visible_a))
        for task in (FAULTY, HEALTHY):
            self.assertFalse((task / 'environment/receipts').exists())
            self.assertFalse((task / 'environment/source_manifest.json').exists())
            self.assertFalse((task / 'environment/validation.json').exists())
            self.assertTrue((task / 'provenance/source_manifest.json').is_file())
        prompt = (FAULTY / 'evaluations/solution/instruction.md').read_text()
        self.assertIn('leave the code unchanged', prompt)
        self.assertNotIn('The incident is already', prompt)
        self.assertNotIn('Implement the needed', (FAULTY / 'workspace/WORKSPACE.md').read_text())

    def test_hidden_case_metadata_and_manifests_match_each_package(self):
        for task, verdict in ((FAULTY, 'unhealthy'), (HEALTHY, 'healthy')):
            metadata = json.loads((task / 'evaluations/solution/verifier/resources/expected_verdict.json').read_text())
            self.assertEqual(metadata['expected_verdict'], verdict)
            manifest = json.loads((task / 'provenance/source_manifest.json').read_text())
            expected = {row['destination']: row for row in manifest['files']}
            current = files(task)
            self.assertEqual(set(expected), set(current) - {'provenance/source_manifest.json'})
            for name, row in expected.items():
                self.assertEqual(hashlib.sha256(current[name].read_bytes()).hexdigest(), row['after_sha256'], name)
                if row['source']:
                    self.assertEqual(hashlib.sha256((ROOT / row['source']).read_bytes()).hexdigest(), row['before_sha256'])
        for name in ('grade_submission.py',):
            self.assertEqual((FAULTY / 'evaluations/solution/verifier/tests' / name).read_bytes(),
                             (HEALTHY / 'evaluations/solution/verifier/tests' / name).read_bytes())
        for name in files(FAULTY / 'workspace/logs'):
            self.assertEqual((FAULTY / 'workspace/logs' / name).read_bytes(), (HEALTHY / 'workspace/logs' / name).read_bytes())

    def test_standard_staging_does_not_expose_author_provenance_or_verdict(self):
        from autorlbench.core.types import EvaluationConfig
        from autorlbench.evaluation.evaluator import TaskEvaluator
        from autorlbench.tasks import load_task_package
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in (FAULTY, HEALTHY):
                package = load_task_package(task)
                evaluator = TaskEvaluator('local')
                config = EvaluationConfig(backend='local', evaluation_level='solution',
                    work_dir=root / task.name, models_dir=root / 'models', datasets_dir=root / 'datasets')
                staged = evaluator._prepare_run(package, package.paths.evaluations['solution'], config)
                for directory in (staged.visible_task, staged.workspace, staged.environment):
                    names = files(directory)
                    self.assertFalse(any('provenance' in Path(n).parts for n in names))
                    self.assertFalse(any(Path(n).name in {'expected_verdict.json', 'source_manifest.json', 'validation.json', 'packaged-reference-grade.json'} for n in names))
                self.assertEqual(set(files(staged.visible_task)), {'instruction.md'})
                self.assertEqual((staged.workspace / 'loader.py').read_bytes(), (task / 'workspace/loader.py').read_bytes())

    def test_starting_loaders_differ_in_behavior_not_incident_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = synthetic_fixture(Path(temporary) / 'fixture')
            contract = load(FAULTY / 'environment/incident_engine/loader_contract.py')
            broken = load(FAULTY / 'workspace/loader.py')
            healthy = load(HEALTHY / 'workspace/loader.py')
            before = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files(fixture.scene).items()}
            misses = 0
            for row in fixture.rows:
                try:
                    contract.read_resolved(fixture.scene / 'dataset', row, broken.resolve)
                except FileNotFoundError:
                    misses += 1
                payload, _ = contract.read_resolved(fixture.scene / 'dataset', row, healthy.resolve)
                self.assertEqual(payload['id'], row['id'])
            self.assertGreater(misses, 0)
            self.assertEqual(before, {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files(fixture.scene).items()})

    def test_no_change_healthy_report_and_repaired_unhealthy_report_use_same_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = synthetic_fixture(root / 'fixture')
            prepared = root / 'prepared'; prepared.mkdir()
            (prepared / 'incident').symlink_to(fixture.scene, target_is_directory=True)
            for task, expected in ((FAULTY, 'unhealthy'), (HEALTHY, 'healthy')):
                verifier = root / (task.name + '-verifier')
                shutil.copytree(task / 'evaluations/solution/verifier', verifier)
                (verifier / 'resources/snapshot.json').write_text(json.dumps({'scene_sha256': hashlib.sha256((fixture.scene / 'SCENE.json').read_bytes()).hexdigest()}))
                workspace = root / (task.name + '-workspace'); workspace.mkdir()
                shutil.copyfile(task / 'workspace/loader.py', workspace / 'loader.py')
                if expected == 'unhealthy':
                    shutil.copyfile(HEALTHY / 'workspace/loader.py', workspace / 'loader.py')
                initial = (workspace / 'loader.py').read_bytes()
                results = root / (task.name + '-results')
                (results / 'experiment_logs').mkdir(parents=True)
                (results / 'experiment_logs/run-home').symlink_to(fixture.home, target_is_directory=True)
                for reported, accepted in ((expected, True), ('healthy' if expected == 'unhealthy' else 'unhealthy', False)):
                    (workspace / 'report.md').write_text(f'Verdict: {reported}\n\nInspected the current resolver and verified the requested content against the manifest.\n')
                    process = subprocess.run([sys.executable, '-I', '-B', str(verifier / 'tests/grade_submission.py')],
                        env=clean_env(AUTORLBENCH_WORKSPACE=workspace, AUTORLBENCH_PREPARED=prepared, AUTORLBENCH_RESULTS=results),
                        capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0 if accepted else 1, process.stdout + process.stderr)
                    grade = json.loads((results / 'result.json').read_text())
                    self.assertIs(grade['passed'], accepted)
                    self.assertEqual(grade['score'], int(accepted))
                    from autorlbench.evaluation.evaluator import TaskEvaluator
                    from types import SimpleNamespace
                    evaluator = TaskEvaluator('local')
                    merged = evaluator._extract_metrics(results, SimpleNamespace(returncode=process.returncode))
                    self.assertEqual(evaluator._extract_score(merged), float(accepted))
                    self.assertIs(merged['passed'], accepted)
                self.assertEqual((workspace / 'loader.py').read_bytes(), initial)


if __name__ == '__main__': unittest.main()
