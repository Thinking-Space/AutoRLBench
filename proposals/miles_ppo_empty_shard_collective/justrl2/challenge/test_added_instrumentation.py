# SPDX-License-Identifier: Apache-2.0
"""Reviewer-only test demonstrating solver-added observations on native code."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import export_challenge
import run_replay


def test_no_supplied_diagnostics_then_source_added_log_and_repair(tmp_path):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for actual native replay')
    export = tmp_path / 'export'
    export_challenge.export(Path(source), export)
    workspace = export / 'workspace'
    replay = workspace / 'inputs/replay.json'
    raw = tmp_path / 'raw'
    result = run_replay.run_replay(workspace, raw, replay=replay)
    assert result['status'] == 'failed', result
    assert set(p.name for p in raw.iterdir()) == {'config.json', 'result.json', *[f'rank{r}.stdout.log' for r in range(4)]}
    logs = ''.join(p.read_text() for p in raw.glob('*.log'))
    for clue in ('whitening', 'all_masks', 'local_numel', 'Traceback', 'requested_members'):
        assert clue not in logs
    observed = tmp_path / 'observed'
    shutil.copytree(workspace, observed)
    loss = observed / export_challenge.TARGET
    text = loss.read_text()
    text = text.replace('        if all_masks.numel() > 0:\n',
                        '        print("USER_OBSERVATION", dist.get_rank(), all_masks.numel(), dist.get_process_group_ranks(dp_group), flush=True)\n'
                        '        if all_masks.numel() > 0:\n')
    loss.write_text(text)
    out = tmp_path / 'instrumented'
    result = run_replay.run_replay(observed, out, replay=replay)
    assert result['status'] == 'failed', result
    for rank, count in enumerate([2, 0, 149, 180]):
        assert f'USER_OBSERVATION {rank} {count} ' in (out / f'rank{rank}.stdout.log').read_text()
    repaired = tmp_path / 'repaired'
    shutil.copytree(workspace, repaired)
    subprocess.run(['git', 'apply', str(export / 'maintainer/repair.patch')], cwd=repaired, check=True)
    result = run_replay.run_replay(repaired, tmp_path / 'corrected', replay=replay)
    assert result['status'] == 'completed', result
