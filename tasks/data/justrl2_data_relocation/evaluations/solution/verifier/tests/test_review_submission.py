"""Report/joint-score contracts using explicit synthetic proofs and tiny fixtures.

No model, GPU, agent evaluation, or native checkpoint deserialization is used.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
GRADER = HERE / 'grade_submission.py'
RESOURCES = HERE.parent / 'resources'
REPO = Path(__file__).resolve().parents[7]
FAKE_PROOF = {'schema_version': 3, 'score': True, 'errors': [],
              'fixture': 'SYNTHETIC proof, not native training evidence',
              'checkpoint_payloads_rehashed': {'actor': True, 'critic': True},
              'diagnostic': {'reward_mean': 0.0, 'counts': [16, 16, 16, 16]}}
FAKE_ENGINE = '''"""Explicit test-only proof function; never a model implementation."""
import json
from pathlib import Path

def verify_resume(scene, home, fresh=True):
    root = Path(__file__).resolve().parent
    with (root / 'calls.jsonl').open('a') as stream:
        stream.write(json.dumps({'scene': str(scene), 'home': str(home), 'fresh': fresh}) + '\\n')
    state = json.loads((root / 'fixture.json').read_text())
    if state.get('error'):
        raise ValueError(state['error'])
    return state['proof']
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('_review_fixture_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Harness:
    def __init__(self, root):
        self.root = root
        self.wrapper = root / 'verifier/tests/grade_submission.py'
        self.resources = root / 'verifier/resources'
        self.engine = self.resources / 'incident_audit'
        self.results, self.workspace, self.prepared = (root / name for name in ('results', 'workspace', 'prepared'))
        self.wrapper.parent.mkdir(parents=True)
        self.engine.mkdir(parents=True)
        self.results.mkdir()
        self.workspace.mkdir()
        shutil.copyfile(GRADER, self.wrapper)
        (self.engine / 'verify_resume.py').write_text(FAKE_ENGINE)
        self.proof()
        write(self.prepared / 'incident/SCENE.json', {'synthetic_fixture': True})
        write(self.resources / 'snapshot.json', {'scene_sha256': sha(self.prepared / 'incident/SCENE.json')})
        self.expected('unhealthy')

    def expected(self, verdict):
        write(self.resources / 'expected_verdict.json', {'schema_version': 1, 'expected_verdict': verdict})

    def proof(self, value=FAKE_PROOF, error=None):
        write(self.engine / 'fixture.json', {'proof': value, 'error': error})

    def report(self, text, archived=True):
        path = (self.results if archived else self.workspace) / 'report.md'
        path.write_text(text, encoding='utf-8')
        return path

    def run(self, workspace_env=True):
        env = dict(os.environ, AUTORLBENCH_RESULTS=str(self.results), AUTORLBENCH_PREPARED=str(self.prepared))
        if workspace_env:
            env['AUTORLBENCH_WORKSPACE'] = str(self.workspace)
        else:
            env.pop('AUTORLBENCH_WORKSPACE', None)
        process = subprocess.run([sys.executable, '-I', '-B', str(self.wrapper)],
                                 env=env, cwd=self.workspace, capture_output=True, text=True)
        assert (self.results / 'result.json').is_file(), process.stderr
        result = json.loads((self.results / 'result.json').read_text())
        assert type(result['score']) is int and result['score'] in (0, 1)
        assert result['passed'] is (result['score'] == 1)
        assert process.returncode == (0 if result['passed'] else 1), process.stderr
        assert json.loads(process.stdout) == result
        assert json.loads((self.results / 'metrics.json').read_text()) == result
        return result

    def calls(self):
        path = self.engine / 'calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def assert_fresh_call(harness):
    assert harness.calls() == [{'scene': str(harness.prepared / 'incident'),
        'home': str(harness.results / 'experiment_logs/run-home'), 'fresh': True}]


def test_wrong_starting_verdict_does_not_replace_successful_continuation_metrics(harness):
    harness.report('Verdict: healthy\nSYNTHETIC evidence about the initial resolver.\n')
    result = harness.run()
    assert result['score'] == 0
    assert result['metrics'] == FAKE_PROOF
    assert result['assessment']['reported'] == 'healthy'
    assert result['assessment']['expected'] == 'unhealthy'
    assert result['assessment']['verdict_correct'] is False
    assert_fresh_call(harness)


@pytest.mark.parametrize('expected,reported,accepted', [
    ('unhealthy', 'healthy', False), ('healthy', 'unhealthy', False),
    ('unhealthy', 'unhealthy', True), ('healthy', 'healthy', True),
    ('unhealthy', None, False), ('healthy', None, False)])
def test_actual_task_evaluator_preserves_joint_score_after_reading_both_json_files(harness, expected, reported, accepted):
    from types import SimpleNamespace
    from autorlbench.evaluation.evaluator import TaskEvaluator

    harness.expected(expected)
    if reported is not None:
        harness.report(f'Verdict: {reported}\nSYNTHETIC starting-state evidence.\n')
    result = harness.run()
    assert result['metrics'] == FAKE_PROOF
    assert result['metrics']['score'] is True
    # Exercise the actual extraction methods without constructing any runtime.
    evaluator = object.__new__(TaskEvaluator)
    merged = evaluator._extract_metrics(harness.results, SimpleNamespace(returncode=0 if accepted else 1))
    assert evaluator._extract_score(merged) == float(accepted)
    assert merged['passed'] is accepted
    assert type(merged['score']) is int and merged['score'] == int(accepted)
    assert merged['assessment'] == result['assessment']
    assert_fresh_call(harness)


@pytest.mark.parametrize('expected', ['healthy', 'unhealthy'])
@pytest.mark.parametrize('reported', ['healthy', 'unhealthy'])
@pytest.mark.parametrize('continued', [False, True])
def test_joint_score_requires_both_correct_starting_verdict_and_fresh_continuation(harness, expected, reported, continued):
    harness.expected(expected)
    proof = {**FAKE_PROOF, 'score': continued}
    harness.proof(proof)
    harness.report(f'\n \t\nVerdict: {reported}\n\nSYNTHETIC evidence and chosen action.\n')
    result = harness.run()
    assert result['score'] == int(expected == reported and continued)
    assert result['assessment']['reported'] == reported
    assert result['assessment']['expected'] == expected
    assert result['assessment']['verdict_correct'] is (expected == reported)
    assert result['assessment']['report_valid'] is True
    assert result['metrics'] == proof
    assert_fresh_call(harness)


@pytest.mark.parametrize('report', [None, '', ' \n\t', 'Verdict: unhealthy', 'Verdict: unhealthy\n \t\n',
    '# Review\nVerdict: unhealthy\nEvidence', 'Verdict: Healthy\nEvidence', 'Verdict: broken\nEvidence',
    'Verdict: unhealthy after repair\nEvidence', '**Verdict: unhealthy**\nEvidence', 'Evidence only'])
def test_missing_or_invalid_report_fails_even_when_continuation_passes(harness, report):
    if report is not None:
        harness.report(report)
    result = harness.run()
    assert result['score'] == 0
    assert result['assessment']['report_valid'] is False
    assert result['assessment']['errors']
    assert result['metrics'] == FAKE_PROOF
    assert_fresh_call(harness)


def test_valid_verdict_without_evidence_is_not_a_valid_report(harness):
    harness.report('Verdict: unhealthy\n\n')
    result = harness.run()
    assert result['assessment']['reported'] == result['assessment']['expected'] == 'unhealthy'
    assert result['assessment']['verdict_correct'] is True
    assert result['assessment']['report_valid'] is False
    assert result['score'] == 0


def test_archived_report_has_priority_over_live_workspace_report(harness):
    archived = harness.report('Verdict: unhealthy\nSYNTHETIC archived evidence.\n')
    harness.report('Verdict: healthy\nSYNTHETIC later workspace evidence.\n', archived=False)
    result = harness.run()
    assert result['score'] == 1
    assert result['assessment']['report_path'] == str(archived)


@pytest.mark.parametrize('archived', ['', 'Verdict: healthy\nSYNTHETIC wrong archived assessment.\n'])
def test_invalid_archived_report_does_not_fall_back_to_better_workspace_report(harness, archived):
    harness.report(archived)
    harness.report('Verdict: unhealthy\nSYNTHETIC live evidence.\n', archived=False)
    result = harness.run()
    assert result['score'] == 0
    assert result['assessment']['report_path'] == str(harness.results / 'report.md')


@pytest.mark.parametrize('workspace_env', [False, True])
def test_standalone_workspace_report_is_used_only_without_archive(harness, workspace_env):
    path = harness.report('Verdict: unhealthy\nSYNTHETIC standalone evidence.\n', archived=False)
    result = harness.run(workspace_env=workspace_env)
    assert result['score'] == 1
    assert result['assessment']['report_path'] == str(path)
    assert_fresh_call(harness)


@pytest.mark.parametrize('damage', ['encoding', 'directory'])
def test_unreadable_report_returns_zero_score_envelope(harness, damage):
    path = harness.results / 'report.md'
    if damage == 'encoding':
        path.write_bytes(b'\xff\xfeSYNTHETIC invalid UTF-8')
    else:
        path.mkdir()
    result = harness.run()
    assert result['score'] == 0 and result['assessment']['errors']
    assert result['metrics'] == FAKE_PROOF


@pytest.mark.parametrize('metadata', [None, 'invalid-json', [], {},
    {'schema_version': 2, 'expected_verdict': 'unhealthy'},
    {'schema_version': True, 'expected_verdict': 'unhealthy'},
    {'schema_version': 1, 'expected_verdict': 'unknown'},
    {'schema_version': 1, 'expected_verdict': True}])
def test_missing_or_invalid_hidden_expectation_fails_robustly(harness, metadata):
    path = harness.resources / 'expected_verdict.json'
    if metadata is None:
        path.unlink()
    elif metadata == 'invalid-json':
        path.write_text('{')
    else:
        write(path, metadata)
    harness.report('Verdict: unhealthy\nSYNTHETIC complete report.\n')
    result = harness.run()
    assert result['score'] == 0
    assert result['assessment']['expected'] is None
    assert result['assessment']['verdict_correct'] is False
    assert result['assessment']['errors']
    assert result['metrics'] == FAKE_PROOF
    assert_fresh_call(harness)


@pytest.mark.parametrize('damage', ['missing_scene', 'scene_pin', 'audit_exception'])
def test_correct_report_never_overrides_missing_or_invalid_continuation(harness, damage):
    harness.report('Verdict: unhealthy\nSYNTHETIC valid starting assessment.\n')
    if damage == 'missing_scene':
        (harness.prepared / 'incident/SCENE.json').unlink()
    elif damage == 'scene_pin':
        write(harness.resources / 'snapshot.json', {'scene_sha256': '0' * 64})
    else:
        harness.proof(error='SYNTHETIC missing checkpoint payload')
    result = harness.run()
    assert result['score'] == 0 and result['metrics']['score'] is False
    assert result['metrics']['errors']
    assert result['assessment']['verdict_correct'] is True
    assert len(harness.calls()) == int(damage == 'audit_exception')


@pytest.mark.parametrize('score', [1, 'true'])
def test_continuation_success_must_be_boolean_true(harness, score):
    proof = {**FAKE_PROOF, 'score': score}
    harness.proof(proof)
    harness.report('Verdict: unhealthy\nSYNTHETIC valid starting assessment.\n')
    result = harness.run()
    assert result['score'] == 0 and result['metrics'] == proof


def test_original_hidden_expected_verdict_is_unhealthy():
    path = RESOURCES / 'expected_verdict.json'
    assert path.is_file(), 'Hidden expected verdict metadata is missing'
    assert json.loads(path.read_text()) == {'schema_version': 1, 'expected_verdict': 'unhealthy'}


def install_real_synthetic_auditor(harness):
    """Use frozen stdlib auditing with explicitly unloadable checkpoint fixtures."""
    fixtures = load(REPO / 'proposals/justrl2_data_relocation/incident/test_verify_resume.py')
    case = fixtures.Fixture(harness.root / 'synthetic_native_contract')
    for name in ('verify_resume.py', 'accounting.py', 'scene.py'):
        shutil.copyfile(RESOURCES / 'incident_audit' / name, harness.engine / name)
    shutil.rmtree(harness.prepared / 'incident')
    (harness.prepared / 'incident').symlink_to(case.scene, target_is_directory=True)
    (harness.results / 'experiment_logs').mkdir()
    (harness.results / 'experiment_logs/run-home').symlink_to(case.home, target_is_directory=True)
    write(harness.resources / 'snapshot.json', {'scene_sha256': sha(case.scene / 'SCENE.json')})
    return case


@pytest.mark.parametrize('expected,reported,changed,accepted', [
    ('healthy', 'healthy', False, True), ('unhealthy', 'unhealthy', True, True),
    ('unhealthy', 'healthy', True, False), ('healthy', 'unhealthy', False, False)])
def test_real_auditor_joint_contract_and_no_change_submission(harness, expected, reported, changed, accepted):
    case = install_real_synthetic_auditor(harness)
    loader = case.home / 'submitted/loader.py'
    before = loader.read_bytes()
    if changed:
        loader.write_text('# SYNTHETIC repaired resolver fixture, never executed\n')
        case.edit('INPUTS.json', lambda value: value.update(loader_sha256=sha(loader)))
    harness.expected(expected)
    harness.report(f'Verdict: {reported}\nSYNTHETIC evidence concerns the starting implementation and chosen action.\n')
    result = harness.run()
    assert result['score'] == int(accepted)
    assert result['metrics']['score'] is True and result['metrics']['admitted_count'] == 64
    assert result['metrics']['checkpoint_payloads_rehashed'] == {'actor': True, 'critic': True}
    assert (loader.read_bytes() != before) is changed
    assert result['assessment']['reported'] == reported


@pytest.mark.parametrize('expected', ['healthy', 'unhealthy'])
def test_real_auditor_still_requires_fresh_checkpoint_files(harness, expected):
    case = install_real_synthetic_auditor(harness)
    harness.expected(expected)
    harness.report(f'Verdict: {expected}\nSYNTHETIC valid starting diagnosis.\n')
    (case.home / 'run/actor/iter_0000007/__0_0.distcp').unlink()
    result = harness.run()
    assert result['assessment']['verdict_correct'] is True
    assert result['metrics']['score'] is False and result['metrics']['errors']
    assert result['score'] == 0
