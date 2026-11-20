import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import export_runtime
import verify_runtime


@pytest.fixture
def exported(tmp_path):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for real candidate verifier integration')
    out = tmp_path / 'export'
    export_runtime.export(Path(source), out)
    return out


def test_repaired_source_passes_independent_numerics(exported, tmp_path):
    workspace = exported / 'workspace'
    subprocess.run(['git', 'apply', str(exported / 'maintainer/repair.patch')], cwd=workspace, check=True)
    result = verify_runtime.verify(workspace, tmp_path / 'result')
    assert result['passed'], result
    assert len(result['cases']) == len(verify_runtime.CASES)
    directory = tmp_path / 'result' / 'long'
    for path in directory.glob('*.trace.jsonl'):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        requests = [e for e in events if e['operation'] == 'reduce_sum']
        for i, event in enumerate(requests):
            event['sequence'] = i
        path.write_text(''.join(json.dumps(e) + '\n' for e in requests))
    corrupted = verify_runtime.assess(directory)
    assert not corrupted['numerical_passed']
    assert any('collective coverage' in e for e in corrupted['failures'])


def test_uncached_import_alias_is_a_valid_repair(exported, tmp_path):
    workspace = exported / 'workspace'
    (workspace / export_runtime.PLAN).write_text(
        'from torch.distributed import all_reduce\ndef clear(): pass\ndef reduce_sum(tensor, group=None):\n    return all_reduce(tensor, group=group)\n')
    result = verify_runtime.verify(workspace, tmp_path / 'result', ['long', 'cold_short', 'warm_short'])
    assert result['passed'], result


def test_original_cache_has_history_dependent_actual_misrouting(exported, tmp_path):
    result = verify_runtime.verify(exported / 'workspace', tmp_path / 'result', ['long', 'cold_short', 'warm_short', 'unequal_dp3'])
    assert not result['passed']
    assert result['cases']['long']['numerical_passed']
    assert result['cases']['unequal_dp3']['numerical_passed']
    for name, phase in [('cold_short', 'whiten'), ('warm_short', 'gae')]:
        case = result['cases'][name]
        assert case['misroutes'] and case['genuine_misroute_timeouts'], case
        assert case['misroutes'][0]['phase'] == phase


@pytest.mark.parametrize('mutation', ['disable_normalization', 'wrong_normalization_group'])
def test_liveness_only_fixes_fail_numerically(exported, tmp_path, mutation):
    workspace = exported / 'workspace'
    subprocess.run(['git', 'apply', str(exported / 'maintainer/repair.patch')], cwd=workspace, check=True)
    path = workspace / 'miles/backends/training_utils/loss.py'
    text = path.read_text()
    if mutation == 'disable_normalization':
        text = text.replace('if args.normalize_advantages:', 'if False:')
    else:
        text = text.replace('dp_group = parallel_state.intra_dp.group', 'dp_group = parallel_state.cp.group')
    path.write_text(text)
    result = verify_runtime.verify(workspace, tmp_path / 'result', ['long'])
    assert not result['passed']
    assert result['cases']['long']['completed']
    assert any('scalar reference' in error for error in result['cases']['long']['failures'])
