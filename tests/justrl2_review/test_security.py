"""Thirteen security and compatibility regressions; temporary markers only, no model runs."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from tests.justrl2_review.fixtures import FAULTY, Fixture, REFERENCE_LOADER, ReviewRun, load

ENGINE = FAULTY / 'environment/incident_engine'


def query(tmp_path, source):
    dataset = tmp_path / 'dataset'
    dataset.mkdir(exist_ok=True)
    script = tmp_path / 'loader.py'
    script.write_text(source)
    request = {'root': str(dataset), 'record': {'id': 'wanted', 'sha256': 'abc', 'path': 'missing.json'}}
    result = subprocess.run([sys.executable, '-I', '-S', str(ENGINE / 'resolve_worker.py'), str(script)],
        input=json.dumps(request), text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_reference_query_preserves_identity_on_the_real_interface(tmp_path):
    case = Fixture(tmp_path / 'case')
    worker = load(ENGINE / 'resolve_worker.py')
    contract = load(ENGINE / 'loader_contract.py')
    source = REFERENCE_LOADER.read_text()
    for row in case.rows:
        payload, _ = contract.read_resolved(case.scene / 'dataset', row,
            lambda root, record: worker.Query(root).run(source, record))
        assert payload['id'] == row['id']


@pytest.mark.parametrize('attack', ['write', 'import', 'introspection', 'traversal'])
def test_query_rejects_code_execution_and_outside_reads(tmp_path, attack):
    marker = tmp_path / 'outside.txt'
    marker.write_text('unchanged')
    lines = {
        'write': 'from pathlib import Path\ndef resolve(root, record):\n    Path(' + repr(str(marker)) + ').write_text("changed")\n    return record["path"]\n',
        'import': 'import os\ndef resolve(root, record):\n    return os.environ["SECRET"]\n',
        'introspection': 'def resolve(root, record):\n    return record.__class__.__mro__\n',
        'traversal': 'from pathlib import Path\ndef resolve(root, record):\n    return (Path(root) / "../outside.txt").read_text()\n',
    }
    result = query(tmp_path, lines[attack])
    assert result['ok'] is False and result['kind'] == 'error'
    assert marker.read_text() == 'unchanged'


def test_query_bounds_nonterminating_and_oversized_work(tmp_path):
    result = query(tmp_path, 'def resolve(root, record):\n    while True:\n        pass\n')
    assert result['ok'] is False and 'Unsupported statement' in result['message']
    worker = load(ENGINE / 'resolve_worker.py')
    with pytest.raises(worker.QueryError, match='large'):
        worker.Query(tmp_path).run('#' * 65537, {})


def test_trusted_launcher_ignores_workspace_python_startup(tmp_path):
    marker = tmp_path / 'executed'
    (tmp_path / 'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("bad")\n')
    env = dict(__import__('os').environ, PYTHONPATH=str(tmp_path), PYTHONUSERBASE=str(tmp_path))
    result = subprocess.run([sys.executable, '-I', '-S', str(ENGINE / 'launch.py'), '--help'],cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_report_symlink_cannot_read_hidden_verdict(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    hidden = tmp_path / 'hidden-report';hidden.write_text('Verdict: healthy\nsecret\n')
    (run.results / 'report.md').symlink_to(hidden)
    result, _ = run.run()
    assert result['score'] == 0 and result['assessment']['report_valid'] is False
    assert 'secret' not in json.dumps(result)


def test_output_symlink_cannot_overwrite_another_file(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    run.submit('Verdict: healthy\nSYNTHETIC checked data.\n')
    marker = tmp_path / 'outside';marker.write_text('unchanged')
    (run.results / 'result.json').symlink_to(marker)
    result, _ = run.run()
    assert result['score'] == 1 and marker.read_text() == 'unchanged'
    assert not (run.results / 'result.json').is_symlink()


def test_failed_verifier_cannot_reuse_a_preplanted_success(tmp_path):
    from autorlbench.evaluation.evaluator import TaskEvaluator
    (tmp_path / 'result.json').write_text('{"passed":true,"score":1}')
    evaluator = object.__new__(TaskEvaluator)
    result = evaluator._extract_metrics(tmp_path, SimpleNamespace(returncode=1), require_success=True)
    assert result['passed'] is False and result['score'] == 0


def test_unsafe_shared_backends_are_rejected_before_execution(tmp_path):
    from autorlbench.evaluation.evaluator import TaskEvaluator
    from autorlbench.tasks.loader import SchemaError, load_task_package
    from autorlbench.evaluation.harbor_evaluator import HarborTaskMaterializer, HarborAdapterError
    package = load_task_package(FAULTY)
    with pytest.raises(SchemaError, match='separate Docker'):
        TaskEvaluator('local').evaluate(FAULTY)
    unit = SimpleNamespace(package=package, evaluation=package.paths.evaluations['solution'])
    materializer = object.__new__(HarborTaskMaterializer)
    with pytest.raises(HarborAdapterError, match='separate Docker'):
        materializer._materialize_one(unit, tmp_path / 'harbor', None)


def test_decoded_json_collections_obey_item_limit(tmp_path):
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    (dataset / 'many.json').write_text(json.dumps([0] * 513))
    result = query(tmp_path, 'import json\nfrom pathlib import Path\ndef resolve(root, record):\n    values = json.loads((Path(root) / "many.json").read_text())\n    return str(len(values))\n')
    assert result['ok'] is False and 'collection limit' in result['message']


def test_other_tasks_keep_legitimate_partial_credit(tmp_path):
    from autorlbench.evaluation.evaluator import TaskEvaluator
    (tmp_path / 'result.json').write_text('{"passed":false,"score":0.6}')
    evaluator = object.__new__(TaskEvaluator)
    result = evaluator._extract_metrics(tmp_path, SimpleNamespace(returncode=1))
    assert result['passed'] is False and result['score'] == 0.6
