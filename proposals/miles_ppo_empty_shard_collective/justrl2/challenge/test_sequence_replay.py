# SPDX-License-Identifier: Apache-2.0
import json
import copy
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
import export_challenge
import sequence_replay
import sequence_audit

HERE = Path(__file__).resolve().parent


def test_schedule_changes_assignments_not_global_inputs():
    config = sequence_replay.default_config()
    steps = [sequence_replay.assignment(config, i) for i in range(5)]
    assert all(sorted(x for batch in step for x in batch) == [0, 1, 2, 3] for step in steps)
    assert steps[:3] == [[[3, 0], [2, 1]], [[2, 0], [3, 1]], [[0, 1], [3, 2]]]
    assert sequence_replay.assignment({**config, 'seed': 4}, 0) != steps[0]
    # The same four source rows are repartitioned; the fixture itself does not change.
    assert {tuple(sorted(i for batch in step for i in batch)) for step in steps} == {(0, 1, 2, 3)}
    assert steps[0] != steps[2]
    with pytest.raises(ValueError):
        sequence_replay.validate_config({**config, 'copies': 1})


def test_metric_aggregation_uses_full_dp_token_count_not_cp_local_numel():
    ranks = []
    for _dp in range(2):
        for cp_rank in range(2):
            rows = []
            for length in (2, 329):
                values = torch.full((length,), 0.25)
                mask = torch.ones(length)
                rows.append((values[cp_rank::2], mask[cp_rank::2], mask.sum()))
            ranks.append(rows)
    contributions = [sequence_replay.metric_contribution(rows, cp_size=2) for rows in ranks]
    assert all(row[0].item() == 2 for row in contributions)
    assert sequence_replay.native_report_reference(contributions, cp_size=2) == pytest.approx([0.25])
    local_counts = [sum(mask.numel() for _, mask, _ in rows) for rows in ranks]
    assert len(set(local_counts)) > 1
    wrong = [
        torch.tensor([
            sum(mask.numel() for _, mask, _ in rows),
            sum(float((values * mask).sum()) for values, mask, _ in rows),
        ])
        for rows in ranks
    ]
    assert sequence_replay.native_report_reference(wrong, cp_size=2) == pytest.approx([0.5])
    assert sequence_replay.native_report_reference(wrong, cp_size=2) != pytest.approx([0.25])


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    if not os.environ.get('JUSTRL2_SOURCE'):
        pytest.skip('set JUSTRL2_SOURCE for real native replay')
    root = tmp_path_factory.mktemp('sequence')
    export_challenge.export(Path(os.environ['JUSTRL2_SOURCE']), root / 'bundle')
    bad = root / 'bundle/workspace'
    good = root / 'repaired'
    shutil.copytree(bad, good)
    subprocess.run(['git', 'apply', str(root / 'bundle/maintainer/repair.patch')], cwd=good, check=True)
    return root, bad, good


def test_same_guard_first_surfaces_as_downstream_error(pair):
    root, bad, good = pair
    config = sequence_replay.default_config()
    before = sequence_replay.run_sequence(bad, root / 'before', config)
    assert before['status'] == 'failed', before
    assert before['completed_updates'] == 2
    assert before['first_error']['operation'] == 'aggregate_train_losses'
    assert before['first_error']['update'] == 2
    assert before['first_error']['rank'] != 3
    assert 'timed out' in before['first_error']['message'].lower()
    input_rows = sequence_replay.inputs(bad)
    before_audit = sequence_audit.validate_sequence(
        config, input_rows, before['evidence'], before['errors'], before['status'])
    assert before_audit['passed'], before_audit
    history = json.loads((root / 'before/history.json').read_text())
    assert len(history) == 2 and all(row['input_summary'] == history[0]['input_summary'] for row in history)
    after = sequence_replay.run_sequence(good, root / 'after', config)
    assert after['status'] == 'completed', after
    assert after['completed_updates'] == config['updates']
    assert after['first_error'] is None
    after_audit = sequence_audit.validate_sequence(
        config, input_rows, after['evidence'], after['errors'], after['status'])
    assert after_audit['passed'], after_audit
    # Judge serialized artifacts too: JSON turns update keys into strings.
    assert sequence_audit.validate_sequence(config, input_rows,
        json.loads(json.dumps(after['evidence'])), after['errors'], 'completed')['passed']
    wrong = copy.deepcopy(after['evidence'])
    wrong[0]['0']['advantages'][0][0] += 1.0
    assert not sequence_audit.validate_sequence(config, input_rows, wrong, [], 'completed')['passed']
    wrong = copy.deepcopy(after['evidence'])
    wrong[0]['0']['whitening_members'] = [0, 1]
    assert not sequence_audit.validate_sequence(config, input_rows, wrong, [], 'completed')['passed']
    wrong = copy.deepcopy(before['errors'])
    wrong = [e for e in wrong if e['operation'] != 'compute_advantages_and_returns']
    assert not sequence_audit.validate_sequence(config, input_rows, before['evidence'], wrong, 'failed')['passed']
    wrong = copy.deepcopy(before['evidence'])
    del wrong[0]['0']
    assert not sequence_audit.validate_sequence(config, input_rows, wrong, before['errors'], 'failed')['passed']
    before_history = json.loads((root / 'before/history.json').read_text())
    after_history = json.loads((root / 'after/history.json').read_text())
    assert [x['metrics'] for x in after_history[:2]] == [x['metrics'] for x in before_history]
    assert set(p.name for p in (root / 'before').iterdir()) == {'summary.json', 'history.json', 'config.json', *[f'rank{r}.stdout.log' for r in range(4)]}


@pytest.mark.parametrize('change,completed', [
    ({'assignment': 'interleave'}, True),
    ({'cp': 1, 'order': [0, 1]}, True),
    ({'reporting': False}, False),
    ({'report_timeout': 7.0}, False),
    ({'compute_timeout': 7.0}, False),
    ({'report_timeout': 4.0, 'compute_timeout': 4.0}, False),
    ({'order': [0, 2, 1, 3]}, False),
])
def test_controls_separate_trigger_from_reporter(pair, change, completed):
    root, bad, good = pair
    config = {**sequence_replay.default_config(), **change}
    label = str(abs(hash(json.dumps(change, sort_keys=True))))
    result = sequence_replay.run_sequence(bad, root / label, config)
    assert (result['status'] == 'completed') is completed, result
    assert result['status'] != 'infrastructure_error', result
    audit = sequence_audit.validate_sequence(
        config, sequence_replay.inputs(bad), result['evidence'], result['errors'], result['status'])
    assert audit['passed'], audit
    if change.get('report_timeout') == 7.0 or change.get('reporting') is False:
        assert result['first_error']['operation'] == 'compute_advantages_and_returns', result
    if change.get('compute_timeout') == 7.0:
        assert result['first_error']['operation'] == 'aggregate_train_losses', result
    if ('report_timeout' in change and 'compute_timeout' in change
            and change['report_timeout'] == change['compute_timeout']):
        assert result['first_error']['operation'] in ('aggregate_train_losses', 'compute_advantages_and_returns'), result



@pytest.mark.parametrize('report_timeout,compute_timeout,accepted_first', [
    (1.0, 4.0, {'aggregate_train_losses'}),
    (7.0, 4.0, {'compute_advantages_and_returns'}),
    (4.0, 4.0, {'aggregate_train_losses', 'compute_advantages_and_returns'}),
])
def test_timeout_controls_change_reported_stage_not_root_failure(pair, tmp_path, report_timeout, compute_timeout, accepted_first):
    _root, bad, _good = pair
    config = {**sequence_replay.default_config(), 'report_timeout': report_timeout,
              'compute_timeout': compute_timeout}
    result = sequence_replay.run_sequence(bad, tmp_path / f'timing-{report_timeout}-{compute_timeout}', config)
    assert result['status'] == 'failed', result
    assert result['completed_updates'] == 2
    assert result['first_error']['operation'] in accepted_first
