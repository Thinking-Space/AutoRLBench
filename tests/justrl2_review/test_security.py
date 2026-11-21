"""Bounded AST evaluation and trusted entry points retain their isolation."""
import json
from types import SimpleNamespace
import sys

import pytest

from tests.justrl2_review.fixtures import ENGINE, TASK, clean_env, execute, load


def query(tmp_path, source):
    dataset = tmp_path / 'dataset'
    dataset.mkdir(exist_ok=True)
    loader = tmp_path / 'loader.py'
    loader.write_text(source, encoding='utf-8')
    request = {'root': str(dataset), 'record': {'id': 'wanted', 'sha256': 'abc', 'path': 'missing.json'}}
    process = execute([sys.executable, '-I', '-S', '-B', ENGINE / 'resolve_worker.py', loader],
                      input=json.dumps(request))
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize('attack', ['write', 'import', 'introspection', 'traversal'])
def test_query_rejects_code_execution_and_outside_reads(tmp_path, attack):
    marker = tmp_path / 'outside.txt'
    marker.write_text('unchanged')
    sources = {
        'write': f'from pathlib import Path\ndef resolve(root, record):\n    Path({str(marker)!r}).write_text("changed")\n    return record["path"]\n',
        'import': 'import os\ndef resolve(root, record):\n    return os.environ["SECRET"]\n',
        'introspection': 'def resolve(root, record):\n    return record.__class__.__mro__\n',
        'traversal': 'from pathlib import Path\ndef resolve(root, record):\n    return (Path(root) / "../outside.txt").read_text()\n',
    }
    result = query(tmp_path, sources[attack])
    assert result['ok'] is False and result['kind'] == 'error'
    assert marker.read_text() == 'unchanged'


def test_query_bounds_nonterminating_and_oversized_work(tmp_path):
    result = query(tmp_path, 'def resolve(root, record):\n    while True:\n        pass\n')
    assert result['ok'] is False and 'Unsupported statement' in result['message']
    worker = load(ENGINE / 'resolve_worker.py')
    with pytest.raises(worker.QueryError, match='large'):
        worker.Query(tmp_path).run('#' * 65537, {})


def test_decoded_json_collections_obey_item_limit(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'many.json').write_text(json.dumps([0] * 513))
    result = query(tmp_path, '''import json
from pathlib import Path
def resolve(root, record):
    values = json.loads((Path(root) / 'many.json').read_text())
    return str(len(values))
''')
    assert result['ok'] is False and 'collection limit' in result['message']


def test_trusted_check_entry_ignores_workspace_python_startup(tmp_path):
    marker = tmp_path / 'executed'
    malicious = f'from pathlib import Path\nPath({str(marker)!r}).write_text("bad")\n'
    (tmp_path / 'sitecustomize.py').write_text(malicious)
    (tmp_path / 'dataset.py').write_text(malicious)
    env = clean_env(PYTHONPATH=tmp_path, PYTHONUSERBASE=tmp_path)
    process = execute([sys.executable, '-I', '-S', '-B', ENGINE / 'check_reads.py', '--help'],
                      env=env, cwd=tmp_path)
    assert process.returncode == 0, process.stderr
    assert not marker.exists()


def test_cpu_task_still_requires_separate_docker_stages(tmp_path):
    from autorlbench.evaluation.evaluator import TaskEvaluator
    from autorlbench.evaluation.harbor_evaluator import HarborAdapterError, HarborTaskMaterializer
    from autorlbench.tasks.loader import SchemaError, load_task_package

    package = load_task_package(TASK)
    assert package.metadata['environment']['gpus'] == 0
    assert package.metadata['security']['requires_isolated_stages'] is True
    with pytest.raises(SchemaError, match='separate Docker'):
        TaskEvaluator('local').rerun(TASK)
    unit = SimpleNamespace(package=package, evaluation=package.paths.evaluations['solution'])
    materializer = object.__new__(HarborTaskMaterializer)
    with pytest.raises(HarborAdapterError, match='separate Docker'):
        materializer._materialize_one(unit, tmp_path / 'harbor', None)
