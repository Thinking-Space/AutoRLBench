# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_runtime


def test_route_transform_preserves_every_native_compute_statement():
    text = 'def f():\n    dist.all_reduce(full_values, group=cp_group)\n    return full_values\n'
    changed = export_runtime.route_reductions(text, 1)
    assert 'reduce_sum(full_values, group=cp_group)' in changed
    assert ast.dump(ast.parse(changed.replace(export_runtime.IMPORT, '').replace('reduce_sum(', 'dist.all_reduce('))) == ast.dump(ast.parse(text))
    with pytest.raises(ValueError):
        export_runtime.route_reductions(text, 2)


def test_reference_repair_only_changes_cache_identity():
    text = (HERE / 'collective_plan.py').read_text()
    fixed = export_runtime.repair_plan(text)
    assert 'key = (group, *_signature(tensor, group))' in fixed
    assert fixed.replace('key = (group, *_signature(tensor, group))', 'key = _signature(tensor, group)') == text


@pytest.mark.skipif(not os.environ.get('JUSTRL2_SOURCE'), reason='pinned source integration')
def test_export_has_no_old_defect_or_old_gpu_evidence(tmp_path):
    out = tmp_path / 'runtime'
    manifest = export_runtime.export(Path(os.environ['JUSTRL2_SOURCE']), out)
    workspace = out / 'workspace'
    loss = workspace / 'miles/backends/training_utils/loss.py'
    assert 'if all_masks.numel() > 0:' not in loss.read_text()
    assert (workspace / 'miles/utils/collective_plan.py').is_file()
    assert (workspace / 'debug/replay_probe.py').is_file()
    assert not (workspace / 'observed').exists()
    assert not (workspace / 'benchmark/train.sh').exists()
    assert (out / 'maintainer/repair.patch').is_file()
    assert manifest['case_kind'] == 'constructed_reduction_plan_cache_regression'
    assert manifest['historical_gpu_evidence_applies'] is False
    assert manifest['old_local_guard_present'] is False
    assert 'source_revision' in manifest
    subprocess.run(['git', 'apply', '--check', str(out / 'maintainer/repair.patch')], cwd=workspace, check=True)
    for file in workspace.rglob('*'):
        if file.is_file():
            assert file.suffix not in ('.patch', '.pyc')
            assert not file.is_symlink()
    with pytest.raises(FileExistsError):
        export_runtime.export(Path(os.environ['JUSTRL2_SOURCE']), out)
