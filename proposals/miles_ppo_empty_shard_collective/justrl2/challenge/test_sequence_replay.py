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
