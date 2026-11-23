# SPDX-License-Identifier: Apache-2.0
"""Real native communicator tests: equal descriptors do not imply identity."""
from datetime import timedelta
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from export_runtime import repair_plan
import plan_identity

HERE = Path(__file__).resolve().parent


@pytest.fixture
def groups(tmp_path):
    dist.init_process_group('gloo', init_method=(tmp_path / 'store').as_uri(),
                            rank=0, world_size=1, timeout=timedelta(seconds=5))
    try:
        yield dist.new_group([0]), dist.new_group([0])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fixed', [False, True])
def test_equal_descriptors_route_to_distinct_native_groups(tmp_path, groups, fixed):
    source = (HERE / 'collective_plan.py').read_text()
    if fixed:
        source = repair_plan(source)
    path = tmp_path / 'candidate.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []

    class RecordingDist:
        def __getattr__(self, name):
            return getattr(dist, name)

        def all_reduce(self, tensor, **kwargs):
            called.append(kwargs['group'])
            return dist.all_reduce(tensor, **kwargs)

    module.dist = RecordingDist()
    first, second = groups
    x = torch.tensor([2.0, 4.0, 6.0])
    assert module._signature(x, first) == module._signature(x, second)
    module.reduce_sum(x, group=first)
    module.reduce_sum(x, group=second)
    assert called[0] is first
    assert called[1] is (second if fixed else first)
    assert torch.equal(x, torch.tensor([2.0, 4.0, 6.0]))
    module.clear()
    module.reduce_sum(x, group=second)
    assert called[-1] is second


@pytest.mark.parametrize('mutation,passes', [('shape', False), ('members', False), ('identity', True), ('late_binding', True), ('import_alias', True)])
def test_candidate_identity_gate_rejects_incomplete_fixes(tmp_path, mutation, passes):
    source = (HERE / 'collective_plan.py').read_text()
    if mutation == 'identity':
        source = repair_plan(source)
    elif mutation == 'shape':
        source = source.replace('key = _signature(tensor, group)', 'key = (tuple(tensor.shape), *_signature(tensor, group))')
    elif mutation == 'members':
        source = source.replace('key = _signature(tensor, group)', 'key = (tuple(dist.get_process_group_ranks(group)), *_signature(tensor, group))')
    elif mutation == 'late_binding':
        source = source.replace('partial(dist.all_reduce, group=group, op=dist.ReduceOp.SUM)',
                                'partial(dist.all_reduce, op=dist.ReduceOp.SUM)').replace('return plan(tensor)', 'return plan(tensor, group=group)')
    else:
        source = 'from torch.distributed import all_reduce\ndef clear(): pass\ndef reduce_sum(tensor, group=None):\n    return all_reduce(tensor, group=group)\n'
    candidate = tmp_path / 'candidate.py'
    candidate.write_text(source)
    result = plan_identity.check(candidate, tmp_path / 'result')
    assert result['passed'] is passes, result
