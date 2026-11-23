# SPDX-License-Identifier: Apache-2.0
"""Reusable execution plans for synchronous sum reductions."""
from functools import partial

import torch.distributed as dist

_plans = {}


def _signature(tensor, group):
    return (
        dist.get_backend(group),
        dist.get_world_size(group),
        dist.get_rank(group),
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tensor.numel(),
    )


def reduce_sum(tensor, group=None):
    if group is None:
        group = dist.group.WORLD
    key = _signature(tensor, group)
    plan = _plans.get(key)
    if plan is None:
        plan = partial(dist.all_reduce, group=group, op=dist.ReduceOp.SUM)
        _plans[key] = plan
    return plan(tensor)


def clear():
    _plans.clear()
