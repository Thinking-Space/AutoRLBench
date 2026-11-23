"""Reusable execution plans for synchronous sum reductions."""
from functools import partial

import torch.distributed as dist

_plans = {}


def _size_class(numel: int) -> str:
    if numel <= 64:
        return "small"
    if numel <= 4096:
        return "medium"
    return "large"


def _signature(tensor, group):
    return (
        dist.get_backend(group),
        dist.get_world_size(group),
        dist.get_rank(group),
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        _size_class(tensor.numel()),
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
