# SPDX-License-Identifier: Apache-2.0
from sequence_audit import response_indices, scalar_sample, validate_sequence


def test_scalar_reference_encodes_pinned_replay_ownership_and_targets():
    short = dict(total=424, length=2, reward=0.0, masks=[1, 1], values=[0.52, 0.52])
    long = dict(total=357, length=329, reward=1.0, masks=[1] * 329, values=[0.52] * 329)
    a = scalar_sample(short, cp=2)
    b = scalar_sample(long, cp=2)
    assert [len(response_indices(short, 2, cp)) for cp in range(2)] == [2, 0]
    assert [len(response_indices(long, 2, cp)) for cp in range(2)] == [149, 180]
    assert a['returns'] == [0.0, 0.0]
    assert b['returns'] == [1.0] * 329


def test_sequence_audit_fails_closed_without_actual_per_sample_rows():
    config = dict(cp=2, dp=2, order=[0, 1, 2, 3], updates=1, copies=2,
                  seed=3, assignment='shuffle', reporting=True,
                  compute_timeout=4.0, report_timeout=1.0, outer_timeout=45.0)
    result = validate_sequence(config, [], [], [], status='failed')
    assert result['passed'] is False
    assert result['failures']
