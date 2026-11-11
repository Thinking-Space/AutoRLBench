"""Contract tests for the contestant boundary and faithful frozen evidence."""
import ast
import copy
import json
import os
import shutil
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_challenge as exporter
from source_cleanup import is_observation

HERE = Path(__file__).resolve().parent.parent


def frozen():
    return json.loads((HERE / 'public_runtime_events.json').read_text())['faulty']


def test_projection_keeps_geometry_but_not_answer_fields():
    files = exporter.project_evidence(frozen(), (HERE / 'public_faulty_trace.txt').read_text(),
                                      json.loads((HERE / 'fixture.json').read_text()))
    text = json.dumps(files)
    for marker in ['whitening', 'loss.py', 'all_reduce', 'local_numel', 'local_advantages',
                   'local_masks', 'mask_sum', 'healthy', 'faulty', 'normalized']:
        assert marker not in text
    mesh = files['mesh.json']
    assert mesh['roles']['actor']['cp_groups'] == [[0, 1], [2, 3]]
    assert mesh['roles']['actor']['dp_groups'] == [[0, 2], [1, 3]]
    progress = files['progress.json']['workers']
    assert next(r for r in progress if r['role'] == 'actor' and r['rank'] == 1)['checkpoints'] == [0, 1]
    assert next(r for r in progress if r['role'] == 'actor' and r['rank'] == 3)['checkpoints'] == [0]
    assert 'wait timeout after 60000ms' in files['failure.txt']
    assert files['replay.json']['samples'][0]['response_length'] == 2


@pytest.mark.parametrize('change', ['missing_rank', 'duplicate', 'wrong_group', 'wrong_length', 'missing_timeout'])
def test_projection_rejects_incomplete_or_inconsistent_evidence(change):
    rows = copy.deepcopy(frozen())
    trace = (HERE / 'public_faulty_trace.txt').read_text()
    if change == 'missing_rank':
        rows = [r for r in rows if (r['role'], r['rank']) != ('critic', 3)]
    elif change == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif change == 'wrong_group':
        rows[0]['dp_members'] = [0, 1]
    elif change == 'wrong_length':
        rows[0]['response_lengths'] = [100]
    else:
        trace = 'outer deadline reached'
    with pytest.raises(ValueError):
        exporter.project_evidence(rows, trace, json.loads((HERE / 'fixture.json').read_text()))


def test_strip_diagnostics_preserves_executable_statements():
    source = '''def f():
    _diag_logger.info("enter whiten %s", 4)
    x = 3
    if x:
        _diag_logger.info("done")
    return x
'''
    cleaned = exporter.strip_diagnostics(source)
    assert '_diag_logger.info' not in cleaned
    ns = {}
    exec(cleaned, ns)
    assert ns['f']() == 3
    ast.parse(cleaned)


def test_native_guard_injection_and_inverse():
    source = '''def compute_advantages_and_returns(args, rollout_data):
    if args.normalize_advantages:
        whitened_advs_flat = distributed_masked_whiten(
            all_advs, all_masks, process_group=dp_group, shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))
    rollout_data["advantages"] = advantages
'''
    changed = exporter.inject_guard(source)
    assert 'if all_masks.numel() > 0:' in changed
    assert exporter.remove_guard(changed) == source
    with pytest.raises(ValueError):
        exporter.inject_guard('def other(): pass\n')


def test_export_real_pinned_source(tmp_path):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE to the pinned public checkout for integration')
    out = tmp_path / 'export'
    receipt = exporter.export(Path(source), out)
    workspace, maintainer = out / 'workspace', out / 'maintainer'
    assert len(list(workspace.glob('miles/**/*.py'))) > 150
    assert (workspace / 'train.py').is_file()
    assert not (workspace / 'benchmark/train.sh').exists()
    assert (maintainer / 'gpu_launch/train.sh').is_file()
    assert 'BENCH_RESTORE_ROOT' not in (maintainer / 'gpu_launch/train.sh').read_text()
    assert 'benchmark.restore_evidence' not in (maintainer / 'gpu_launch/run_training.py').read_text()
    assert (maintainer / 'repair.patch').is_file()
    assert (maintainer / 'manifest.json').is_file()
    assert (maintainer / 'topology_verifier.py').is_file()
    assert (maintainer / 'mesh.json').is_file()
    assert (maintainer / 'progress.json').is_file()
    assert (maintainer / 'failure.txt').is_file()
    assert not (workspace / 'observed').exists()
    assert (workspace / 'inputs/replay.json').is_file()
    assert (workspace / 'debug/run_updates.py').is_file()
    assert (workspace / 'debug/runtime_support.py').is_file()
    assert not (workspace / 'debug/run_replay.py').exists()
    assert '--trace' not in (workspace / 'debug/run_updates.py').read_text()
    assert not (workspace / 'miles/utils/debug_utils/run_megatron').exists()
    assert 'MILES_LOGPROB_BACKWARD_DEBUG_DIR' not in (workspace / 'miles/utils/ppo_utils.py').read_text()
    assert 'torch.save' not in (workspace / 'miles/utils/train_dump_utils.py').read_text()
    assert 'register_non_intrusive_dumper' not in (workspace / 'miles/utils/dumper_utils.py').read_text()
    assert receipt['gpu_revalidated'] is False
    assert receipt['source_revision'] == '75814318f243af7e39af337c1173fc3927c9635b'
    for p in workspace.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():
            assert p.suffix not in ('.patch', '.pyc', '.pt', '.safetensors')
            assert '.git' not in p.parts
            assert 'whitening_enter' not in p.read_text()
            assert 'whitening_call' not in p.read_text()
            if p.suffix == '.py' and p.relative_to(workspace).parts[0] in ('miles', 'miles_plugins'):
                assert not any(is_observation(node) for node in ast.walk(ast.parse(p.read_text()))), p
    target = workspace / 'miles/backends/training_utils/loss.py'
    assert 'if all_masks.numel() > 0:' in target.read_text()
    assert 'Collective-communication invariant: distributed_masked_whiten' not in target.read_text()
    with pytest.raises(FileExistsError):
        exporter.export(Path(source), out)
    second = tmp_path / 'second'
    exporter.export(Path(source), second)
    assert exporter.inventory(workspace) == exporter.inventory(second / 'workspace')
    assert (out / 'workspace.tar.gz').read_bytes() == (second / 'workspace.tar.gz').read_bytes()
    with tarfile.open(out / 'workspace.tar.gz') as archive:
        assert set(archive.getnames()) == set(exporter.inventory(workspace))
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
                   for m in archive.getmembers())


def test_source_pin_rejects_modified_extracted_tree(tmp_path):
    (tmp_path / 'train.py').write_text('print("not the pinned source")\n')
    with pytest.raises(ValueError, match='inventory'):
        exporter.read_source(tmp_path)


def test_source_pin_rejects_external_link(tmp_path):
    (tmp_path / 'train.py').symlink_to(HERE / 'build_case.py')
    with pytest.raises(ValueError, match='symlinks'):
        exporter.read_source(tmp_path)


@pytest.mark.parametrize('name', ['fixture.json', 'public_runtime_events.json', 'public_faulty_trace.txt'])
def test_changed_frozen_evidence_is_rejected(tmp_path, monkeypatch, name):
    for file in ('fixture.json', 'public_runtime_events.json', 'public_faulty_trace.txt', 'public_runtime_results.json'):
        shutil.copyfile(HERE / file, tmp_path / file)
    path = tmp_path / name
    if name == 'fixture.json':
        fixture = json.loads(path.read_text())
        for sample in fixture['samples']:
            sample['loss_mask'] = [0] * sample['response_length']
        path.write_text(json.dumps(fixture))
    else:
        path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(exporter, 'EVIDENCE', tmp_path)
    with pytest.raises(ValueError, match='differs'):
        exporter.read_evidence()


def test_replay_pin_matches_original_construction_receipt():
    checks = json.loads((HERE / 'construction_checks.json').read_text())
    assert exporter.REPLAY_SHA256 == checks['fixture']['files']['replay.json']['sha256']


def test_exported_computation_matches_original_recorded_pair(tmp_path):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for native source equivalence')
    sys.path.insert(0, str(HERE))
    import build_case
    pair = tmp_path / 'pair'
    build_case.build(Path(source), pair)
    out = tmp_path / 'challenge'
    exporter.export(Path(source), out)

    class RemoveObservationCalls(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                call = node.value.func
                if isinstance(call, ast.Name) and call.id == 'emit':
                    return None
                if (isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
                        and call.value.id == '_diag_logger'):
                    return None
            return self.generic_visit(node)

        def visit_Pass(self, node):
            return None

    def computation(text):
        function = next(n for n in ast.parse(text).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantages_and_returns')
        return ast.dump(RemoveObservationCalls().visit(function), include_attributes=False)

    candidate = (out / 'workspace' / exporter.TARGET).read_text()
    assert computation(candidate) == computation((pair / 'faulty' / exporter.TARGET).read_text())
    assert computation(exporter.remove_guard(candidate)) == computation((pair / 'healthy' / exporter.TARGET).read_text())
