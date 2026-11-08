# SPDX-License-Identifier: Apache-2.0
"""Reviewer-only test; the observation line is not exported to solvers."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest
import export_runtime
import run_replay
from source_cleanup import is_observation
import ast


def test_pristine_run_requires_added_source_observations(tmp_path):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for native integration')
    bundle = tmp_path / 'export'
    export_runtime.export(Path(source), bundle)
    workspace = bundle / 'workspace'
    assert not (workspace / 'debug/replay_probe.py').exists()
    runner = (workspace / 'debug/run_replay.py').read_text()
    assert '--trace' not in runner and '_Trace' not in runner
    for root in ('miles', 'miles_plugins'):
        for path in (workspace / root).rglob('*.py'):
            assert not any(is_observation(n) for n in ast.walk(ast.parse(path.read_text()))), path
    raw = tmp_path / 'raw'
    result = run_replay.run_replay(workspace, raw)
    assert result['status'] == 'failed', result
    assert set(p.name for p in raw.iterdir()) == {'config.json', 'result.json', *[f'rank{r}.stdout.log' for r in range(4)]}
    logs = ''.join(p.read_text() for p in raw.glob('*.log'))
    for clue in ('whitening', 'requested_members', 'actual_members', 'Traceback', 'collective_plan'):
        assert clue not in logs
    observed = tmp_path / 'observed'
    shutil.copytree(workspace, observed)
    plan = observed / export_runtime.PLAN
    text = plan.read_text().replace('    return plan(tensor)',
        '    print("USER_ROUTE", dist.get_rank(), dist.get_process_group_ranks(group), dist.get_process_group_ranks(plan.keywords["group"]), flush=True)\n    return plan(tensor)')
    plan.write_text(text)
    output = tmp_path / 'instrumented'
    result = run_replay.run_replay(observed, output)
    assert result['status'] == 'failed', result
    rank0 = (output / 'rank0.stdout.log').read_text()
    assert 'USER_ROUTE 0 [0, 1] [0, 2]' in rank0
    repaired = tmp_path / 'repaired'
    shutil.copytree(workspace, repaired)
    subprocess.run(['git', 'apply', str(bundle / 'maintainer/repair.patch')], cwd=repaired, check=True)
    result = run_replay.run_replay(repaired, tmp_path / 'corrected')
    assert result['status'] == 'completed', result
