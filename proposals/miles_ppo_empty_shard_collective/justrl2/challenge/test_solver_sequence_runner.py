# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import export_challenge

HERE = Path(__file__).resolve().parent


def load_runner(runner_path):
    assert runner_path.is_file(), 'multi-update solver entrypoint not exported'
    import sys
    sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location('solver_run_updates', runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for native replay')
    root = tmp_path_factory.mktemp('candidate-sequence')
    bundle = root / 'bundle'
    export_challenge.export(Path(source), bundle)
    workspace = bundle / 'workspace'
    fixed = root / 'fixed'
    shutil.copytree(workspace, fixed)
    subprocess.run(['git', 'apply', str(bundle / 'maintainer/repair.patch')], cwd=fixed, check=True)
    return root, workspace, fixed


def test_sequence_preserves_global_inputs_and_changes_only_batch_assignment(candidate):
    root, workspace, _fixed = candidate
    runner = load_runner(workspace / 'debug/run_updates.py')
    result = runner.run_updates(workspace, root / 'pristine')
    assert result['status'] == 'failed' and result['exit_code'] == 1
    assert result['completed_updates'] == 2 and result['outer_timeout'] is False
    assert result['first_error']['operation'] == 'aggregate_train_losses'
    assert result['first_error']['update'] == 2
    assert 'timed out' in result['first_error']['message'].lower()
    assert '/Users/' not in result['first_error']['message']
    assert set(p.name for p in (root / 'pristine').iterdir()) == {
        'result.json', *[f'rank{r}.stdout.log' for r in range(4)]
    }
    logs = ''.join(p.read_text() for p in (root / 'pristine').glob('*.log'))
    assert 'aggregate_train_losses' not in logs
    assert 'compute_advantages_and_returns' not in logs
    assert 'Traceback' not in logs


def test_agent_added_observation_finds_variable_local_batch_and_repair_completes(candidate):
    root, workspace, fixed = candidate
    runner = load_runner(workspace / 'debug/run_updates.py')
    observed = root / 'observed'
    shutil.copytree(workspace, observed)
    loss = observed / 'miles/backends/training_utils/loss.py'
    source = loss.read_text()
    anchor = '        if all_masks.numel() > 0:\n'
    assert source.count(anchor) == 1
    loss.write_text(source.replace(
        anchor,
        '        print("LOCAL", dist.get_rank(), all_masks.numel(), flush=True)\n' + anchor,
        1,
    ))
    result = runner.run_updates(observed, root / 'instrumented')
    assert result['status'] == 'failed' and result['completed_updates'] == 2
    logs = ''.join(p.read_text() for p in (root / 'instrumented').glob('*.log'))
    assert 'LOCAL 1 0' in logs and 'LOCAL 3 360' in logs
    repaired = runner.run_updates(fixed, root / 'repaired')
    assert repaired == {'status': 'completed', 'exit_code': 0, 'completed_updates': 5, 'outer_timeout': False, 'first_error': None}
