# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import export_challenge


def load(path):
    import sys
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('window_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    source = os.environ.get('JUSTRL2_SOURCE')
    if not source:
        pytest.skip('set JUSTRL2_SOURCE for native replay')
    root = tmp_path_factory.mktemp('window')
    bundle = root / 'export'
    export_challenge.export(Path(source), bundle)
    workspace = bundle / 'workspace'
    fixed = root / 'fixed'
    shutil.copytree(workspace, fixed)
    subprocess.run(['git', 'apply', str(bundle / 'maintainer/repair.patch')], cwd=fixed, check=True)
    return root, workspace, fixed, bundle


def test_completed_window_has_wrong_policy_loss_and_reference_repairs_it(candidate):
    root, workspace, fixed, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    defective = runner.run_updates(workspace, root / 'faulty')
    repaired = runner.run_updates(fixed, root / 'repaired')
    assert defective['status'] == repaired['status'] == 'completed'
    assert defective['completed_updates'] == repaired['completed_updates'] == 5
    assert defective['first_error'] is repaired['first_error'] is None
    assert defective['validation'] == 'failed' and repaired['validation'] == 'passed'
    a = json.loads((root / 'faulty/metrics.json').read_text())
    b = json.loads((root / 'repaired/metrics.json').read_text())
    for update, (fault, repair) in enumerate(zip(a, b, strict=True)):
        assert all(math.isfinite(v) for v in fault.values())
        assert fault['value_loss'] == repair['value_loss']
        if update < 4:
            assert fault['clip_fraction'] == repair['clip_fraction']
        if update in (3, 4):
            assert abs(fault['policy_loss'] - repair['policy_loss']) > .09
        else:
            assert fault == repair
    assert set(p.name for p in (root / 'faulty').iterdir()) == {
        'result.json', 'metrics.json', *[f'rank{r}.stdout.log' for r in range(4)]}


def test_reporting_frequency_changes_symptom_not_repair(candidate):
    root, workspace, fixed, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    config = dict(runner.default_config(), report_every=1)
    a = runner.run_updates(workspace, root / 'synchronous-fault', config)
    b = runner.run_updates(fixed, root / 'synchronous-fixed', config)
    assert a['status'] == 'completed' and a['completed_updates'] == 5
    assert a['validation'] == 'failed' and a['first_error'] is None
    assert a['outer_timeout'] is False
    assert b['status'] == 'completed' and b['validation'] == 'passed'


def test_mixed_assignment_and_cp1_avoid_the_fault(candidate):
    root, workspace, _, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    for name, overrides in [('mixed', dict(assignment='interleave')),
                            ('cp1', dict(cp=1, order=[0, 1]))]:
        result = runner.run_updates(workspace, root / name, dict(runner.default_config(), **overrides))
        assert result['status'] == 'completed' and result['validation'] == 'passed'


def test_validation_rejects_missing_nonfinite_or_wrong_metrics(candidate):
    _, workspace, _, _bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    config = runner.default_config()
    expected = json.loads((workspace / 'inputs/expected.json').read_text())[str(config['cp'])]
    assert runner.validate_metrics(expected, expected)
    assert not runner.validate_metrics(expected[:-1], expected)
    changed = json.loads(json.dumps(expected))
    changed[0]['policy_loss'] = float('nan')
    assert not runner.validate_metrics(changed, expected)
    changed[0]['policy_loss'] = expected[0]['policy_loss'] + .01
    assert not runner.validate_metrics(changed, expected)


def test_private_audit_detects_stale_statistics_with_aligned_calls(candidate):
    import copy
    import window_probe
    from window_reference import validate_trace
    root, workspace, fixed, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    config = runner.default_config()
    fixture = json.loads((bundle / 'maintainer/window_evaluator/inputs/replay.json').read_text())
    for name, source, poisoned in [('fault-probe', workspace, True), ('fix-probe', fixed, False)]:
        output = root / name
        result = window_probe.run(source, output, config, evaluator=bundle / 'maintainer/window_evaluator')
        assert result['status'] == 'completed'
        traces = {str(r): json.loads((output / f'private-{r}.json').read_text()) for r in range(4)}
        metrics = json.loads((output / 'metrics.json').read_text())
        audited = validate_trace(fixture, config, traces, metrics, poisoned_identity=poisoned)
        assert audited['passed'], audited['errors']
        assert all(x['updates'][0] == x['updates'][1] for x in audited['pairings'])
        assert all(row['whitening_called'] for trace in traces.values() for row in trace['rows'])
        if poisoned:
            assert not validate_trace(fixture, config, traces, metrics)['passed']
            assert traces['1']['rows'][3]['collective_input'][2] > len(traces['1']['rows'][3]['whitening_mask'])
            assert traces['1']['rows'][3]['identity_after'] == traces['1']['rows'][2]['collective_output']
            bad = copy.deepcopy(traces)
            next(part for part in bad['3']['rows'][4]['advantages'] if part)[0] += .01
            assert not validate_trace(fixture, config, bad, metrics, poisoned_identity=True)['passed']
            del bad['1']
            assert not validate_trace(fixture, config, bad, metrics, poisoned_identity=True)['passed']


def test_private_verifier_ignores_candidate_harness_and_grades_policy_loss(candidate):
    import window_probe
    root, workspace, fixed, bundle = candidate
    tampered = root / 'tampered-harness'
    shutil.copytree(workspace, tampered)
    (tampered / 'debug/run_window.py').write_text('raise RuntimeError("candidate harness must not run")\n')
    (tampered / 'debug/runtime_support.py').write_text('raise RuntimeError("candidate loader must not run")\n')
    result = window_probe.run(tampered, root / 'private-tampered', evaluator=bundle / 'maintainer/window_evaluator')
    assert result['status'] == 'completed' and result['independent_validation'] == 'failed'
    assert result['exit_code'] == 1
    result = window_probe.run(fixed, root / 'private-accepted', evaluator=bundle / 'maintainer/window_evaluator')
    assert result['independent_validation'] == 'passed' and result['exit_code'] == 0


def test_longer_deadlines_do_not_change_prefix_correctness(candidate):
    root, workspace, _, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    config = dict(runner.default_config(), updates=3, report_every=1, compute_timeout=23., report_timeout=.2, outer_timeout=40.)
    result = runner.run_updates(workspace, root / 'long-compute', config)
    assert result['status'] == 'completed' and result['outer_timeout'] is False
    assert result['first_error'] is None and result['validation'] == 'passed'


def test_invalid_post_run_result_still_produces_structured_status(candidate):
    root, workspace, fixed, bundle = candidate
    runner = load(workspace / 'debug/run_window.py')
    invalid = root / 'invalid-reference'
    shutil.copytree(fixed, invalid)
    (invalid / 'inputs/expected.json').write_text('{broken json')
    result = runner.run_updates(invalid, root / 'invalid-reference-run')
    assert result['status'] == 'infrastructure_error' and result['exit_code'] == 2
    assert result['first_error']['operation'] == 'validate_outputs'
    assert json.loads((root / 'invalid-reference-run/result.json').read_text()) == result


def test_empty_shard_only_prefix_passes_then_later_updates_expose_stale_moments(candidate):
    import window_probe
    root, workspace, fixed, bundle = candidate
    evaluator = bundle / 'maintainer/window_evaluator'
    fixture = json.loads((evaluator / 'inputs/replay.json').read_text())
    runner = load(workspace / 'debug/run_window.py')
    for updates in (3, 4):
        config = dict(runner.default_config(), updates=updates)
        output = root / f'history-{updates}'
        result = window_probe.run(workspace, output, config, evaluator=evaluator)
        assert result['status'] == 'completed'
        assert result['completed_updates'] == updates
        if updates == 3:
            assert result['validation'] == 'passed'
            assert result['independent_validation'] == 'passed'
        else:
            assert result['independent_validation'] == 'failed'
            assert any('update 3' in item for item in result['independent_errors'])
        # A fresh worker process has a new reducer cache; no cross-run state leaks.
        repaired = window_probe.run(fixed, root / f'history-{updates}-fixed', config, evaluator=evaluator)
        assert repaired['independent_validation'] == 'passed'


def test_actual_source_has_no_rank_local_collective_skip(candidate):
    _, workspace, fixed, _bundle = candidate
    for source in (workspace, fixed):
        loss = (source / 'miles/backends/training_utils/loss.py').read_text()
        assert 'if all_masks.numel() > 0:' not in loss
        reducer = (source / 'miles/utils/moments.py').read_text()
        assert 'reduce(add, blocks, origin)' in reducer
        distributed = (source / 'miles/utils/distributed_utils.py').read_text()
        assert distributed.count('dist.all_reduce(stats_tensor, group=process_group)') == 1


def test_fresh_empty_accumulator_is_an_equivalent_valid_repair(candidate):
    import window_probe
    root, workspace, _reference_fix, bundle = candidate
    evaluator = bundle / 'maintainer/window_evaluator'
    repaired = root / 'fresh-empty-identity-fix'
    shutil.copytree(workspace, repaired)
    moments = repaired / 'miles/utils/moments.py'
    source = moments.read_text()
    anchor = '        blocks = (\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '        if values.numel() == 0:\n            return torch.zeros_like(origin)\n' + anchor, 1)
    moments.write_text(source)
    result = window_probe.run(repaired, root / 'fresh-empty-identity-run', evaluator=evaluator)
    assert result['status'] == 'completed'
    assert result['validation'] == 'passed'
    assert result['independent_validation'] == 'passed'
