import math

from numerical_reference import numeric_equal, reference_step, response_indices


def test_cp_geometry_independent_of_native_source():
    assert [len(response_indices(424, 2, 2, i)) for i in range(2)] == [2, 0]
    assert [len(response_indices(357, 329, 2, i)) for i in range(2)] == [149, 180]
    assert sorted(response_indices(357, 329, 2, 0) + response_indices(357, 329, 2, 1)) == list(range(329))


def test_scalar_gae_and_unbiased_variance():
    samples = [[dict(total_length=5, response_length=2, values=[0.2, 0.4], loss_mask=[1, 1], reward=1.0)],
               [dict(total_length=5, response_length=2, values=[0.3, 0.1], loss_mask=[1, 1], reward=0.0)]]
    rows = reference_step(samples, 1, [0, 1])
    assert numeric_equal(rows[0]['raw'], [[0.2 + math.sqrt(0.5) * 0.6, 0.6]])
    assert rows[0]['returns'] == [[1.0, 1.0]]
    values = rows[0]['advantages'][0] + rows[1]['advantages'][0]
    assert abs(sum(values)) < 1e-12
    assert math.isclose(sum(x*x for x in values)/3, 1.0, abs_tol=1e-6)


def test_numeric_comparison_rejects_nonfinite_and_wrong_shapes():
    assert not numeric_equal([[float('nan')]], [[0.0]])
    assert not numeric_equal([[0.0]], [[0.0, 0.0]])
    assert not numeric_equal([[True]], [[1.0]])
