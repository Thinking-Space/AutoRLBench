# SPDX-License-Identifier: Apache-2.0
"""Local-only benchmark evidence; sampled equality is not proof of no update."""
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

_MASTER_GROUPS = ("shard_fp32_from_float16_groups", "fp32_from_float16_groups",
                  "shard_fp32_groups", "fp32_from_fp32_groups")
_MAX_BLOCKS, _MAX_ELEMENTS = 4096, 512 * 1024  # <=2 MiB per snapshot.


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Evidence accepts only scalar tensors")
        return value.detach().item()
    return value


def emit(args, event, **fields):
    """Append durable JSONL; require initialized real groups and propagate errors."""
    root, role = os.environ.get("BENCH_EVIDENCE_DIR"), getattr(args, "bench_role", None)
    if not root or role not in ("actor", "critic"):
        raise ValueError("BENCH_EVIDENCE_DIR and args.bench_role=actor/critic are required")
    if not dist.is_initialized():
        raise RuntimeError("Evidence requires an initialized distributed worker")
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    record = dict(event=event, role=role, rank=dist.get_rank(),
                  pid=os.getpid(), monotonic=time.monotonic())
    for name, group in (("cp", state.cp), ("dp", state.intra_dp)):
        if group.group is None:
            raise RuntimeError(f"Missing {name} process group")
        record.update({f"{name}_rank": group.rank, f"{name}_size": group.size,
                       f"{name}_members": dist.get_process_group_ranks(group.group)})
    if record.keys() & fields.keys():
        raise ValueError("Evidence fields cannot override worker identity or event")
    record.update(fields)
    line = json.dumps(record, default=_scalar, allow_nan=False, sort_keys=True) + "\n"
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{role}-rank{record['rank']}-pid{record['pid']}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        if stream.write(line) != len(line):
            raise OSError("Short evidence write")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _parameters(optimizer):
    pending, visited, parameters = [optimizer], set(), {}
    while pending:
        opt = pending.pop()
        if opt is None or id(opt) in visited:
            continue
        visited.add(id(opt))
        children = list(getattr(opt, "chained_optimizers", None) or ())
        inner = getattr(opt, "optimizer", None)
        if inner is not None:
            children.append(inner)
        pending.extend(children)
        groups = () if children else (getattr(opt, "param_groups", None) or ())
        tensors = [p for group in groups for p in group["params"]]
        for name in _MASTER_GROUPS:
            for group in getattr(opt, name, None) or ():
                tensors.extend(group)
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Optimizer parameter groups must contain tensors")
            if tensor.dtype == torch.float32 and tensor.numel():
                parameters[id(tensor)] = tensor
                if len(parameters) > _MAX_BLOCKS:
                    raise RuntimeError("Too many FP32 blocks for bounded evidence")
    if not parameters:
        raise RuntimeError("No nonempty FP32 optimizer/master tensors found")
    return parameters


def snapshot_optimizer(optimizer):
    """Opaque process-local snapshot: every FP32 block, <=512 spaced values each."""
    parameters = _parameters(optimizer)
    per_block = min(512, _MAX_ELEMENTS // len(parameters))
    result = {}
    for key, tensor in parameters.items():
        if tensor.layout != torch.strided:
            raise TypeError("Only strided optimizer tensors are supported")
        value, count = tensor.detach(), min(per_block, tensor.numel())
        indices = torch.arange(count, device=value.device, dtype=torch.int64)
        indices = indices * (value.numel() - 1) // max(count - 1, 1)
        if value.is_contiguous():
            sampled = value.view(-1).index_select(0, indices)
        else:
            sampled = value[torch.unravel_index(indices, value.shape)]
        result[key] = (tuple(value.shape), value.numel(), sampled.to("cpu", copy=True))
    return result


def after_optimizer(args, optimizer, before, update_successful, grad_norm, rollout_id, step_id):
    """Record the real step result and sampled changes, without inferring success."""
    success = _scalar(update_successful)
    if not isinstance(success, bool):
        raise TypeError("update_successful must be the actual boolean optimizer result")
    norm = None if grad_norm is None else float(_scalar(grad_norm))
    norm_finite = norm is not None and math.isfinite(norm)
    after = snapshot_optimizer(optimizer)
    if not before or before.keys() != after.keys():
        raise RuntimeError("FP32 parameter identities changed between snapshots")
    changed, finite, max_delta, elements = 0, True, 0.0, 0
    for key, (shape, numel, value) in after.items():
        old_shape, old_numel, old = before[key]
        if (shape, numel, value.shape) != (old_shape, old_numel, old.shape):
            raise RuntimeError("FP32 parameter shape/sample count changed")
        delta = (value - old).abs()
        block_finite = bool(torch.isfinite(value).all() & torch.isfinite(old).all()
                            & torch.isfinite(delta).all())
        finite = finite and block_finite
        if block_finite:
            changed += int(bool((delta != 0).any()))
            max_delta = max(max_delta, float(delta.max()))
        elements += value.numel()
    return emit(args, "optimizer_step", rollout_id=rollout_id, step_id=step_id,
                update_successful=success, grad_norm=norm if norm_finite else None,
                grad_norm_finite=norm_finite, sampled_finite=finite,
                sampled_changed=bool(changed) if finite else None,
                changed_tensors=changed if finite else None,
                max_delta=max_delta if finite else None,
                sampled_tensors=len(after), sampled_elements=elements,
                sampled_bytes=elements * 4)
