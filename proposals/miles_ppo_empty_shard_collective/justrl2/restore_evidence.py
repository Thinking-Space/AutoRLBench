# SPDX-License-Identifier: Apache-2.0
"""Exact, process-independent fingerprints and fail-closed restore diagnostics.

Only trusted local Megatron checkpoints are supported. Tensor hashes include
all bytes (including BF16), never just sampled values. No checkpoint is written
by this module. Imports of Megatron are deferred until running inside a worker.
"""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch

# Process-local only: never attach live models/callables to args, which Megatron
# serializes into common.pt on its next checkpoint save.
_RESTORE_CONTEXT = {}


def _unwrap(value):
    # Megatron's sharded wrappers carry tensors/objects in .data. Avoid importing
    # Megatron for CPU tests or treating a Tensor's .data as another wrapper.
    if type(value).__module__.startswith('megatron.core.dist_checkpointing') and hasattr(value, 'data'):
        return value.data
    return value


def _is_nonpersistent(value):
    return (type(value).__module__.startswith('megatron.core.dist_checkpointing')
            and type(value).__name__ == 'LocalNonpersistentObject')


def _nonpersistent_paths(value, path=()):
    if _is_nonpersistent(value):
        yield path
        return
    value = _unwrap(value)
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    for key, item in items:
        yield from _nonpersistent_paths(item, (*path, str(key)))


def _leaves(value, path=(), excluded=()):
    if path in excluded or _is_nonpersistent(value):
        return
    value = _unwrap(value)
    if isinstance(value, dict):
        for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
            yield from _leaves(value[key], (*path, str(key)), excluded)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _leaves(item, (*path, str(i)), excluded)
    else:
        yield path, value


def _digest(value, excluded=()):
    sha, tensors, elements = hashlib.sha256(), 0, 0
    steps, moments = [], {'exp_avg': 0, 'exp_avg_sq': 0}
    for path, item in _leaves(value, excluded=excluded):
        sha.update(json.dumps(path).encode() + b'\0')
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise ValueError('Unsupported tensor layout in restore state')
            item = item.detach()
            sha.update(json.dumps([str(item.dtype), list(item.shape)]).encode())
            flat = item.reshape(-1)
            # Transfer/hash bounded chunks; no full model CPU clone for hashing.
            for block in flat.split(1024 * 1024):
                cpu = block.to('cpu').contiguous()
                if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all()):
                    raise ValueError('Nonfinite restore state: ' + '.'.join(path))
                sha.update(cpu.view(torch.uint8).numpy().tobytes())
            tensors += 1
            elements += item.numel()
            scalar = item.item() if item.numel() == 1 else None
        elif isinstance(item, (bytes, bytearray)):
            sha.update(bytes(item))
            scalar = None
        elif item is None or isinstance(item, (str, int, float, bool)):
            sha.update(json.dumps(item, allow_nan=False).encode())
            scalar = item
        else:
            raise ValueError('Unsupported restore state leaf: ' + str(type(item)))
        if path and path[-1] == 'step':
            if isinstance(scalar, bool) or not isinstance(scalar, (float, int)) or not math.isfinite(scalar):
                raise ValueError('Invalid Adam step')
            steps.append(float(scalar))
        for key in moments:
            if key in path and isinstance(item, torch.Tensor):
                moments[key] += item.numel()
    return sha.hexdigest(), tensors, elements, sorted(steps), moments


def _canonical_digest(value, excluded=()):
    """Hash persisted leaves independent of local DCP shard path names."""
    leaves = []
    for path, item in _leaves(value, excluded=excluded):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise ValueError('Unsupported tensor layout in canonical restore state')
            item = item.detach()
            raw = bytearray()
            for block in item.reshape(-1).split(1024 * 1024):
                cpu = block.to('cpu').contiguous()
                if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all()):
                    raise ValueError('Nonfinite canonical restore state')
                raw.extend(cpu.view(torch.uint8).numpy().tobytes())
            leaves.append(('tensor', str(item.dtype), tuple(item.shape), bytes(raw)))
        elif isinstance(item, (bytes, bytearray)):
            leaves.append(('bytes', bytes(item)))
        else:
            leaves.append(('scalar', json.dumps(item, allow_nan=False, sort_keys=True)))
    leaves.sort(key=lambda item: repr(item))
    sha = hashlib.sha256()
    for item in leaves:
        sha.update(repr(item).encode())
        sha.update(b'\0')
    return sha.hexdigest()


def fingerprint(state, *, excluded=()):
    models = {k: v for k, v in state.items() if k == 'model' or k.startswith('model') and k[5:].isdigit()}
    if not models or 'optimizer' not in state:
        raise ValueError('Both model and optimizer state are required')
    model = _digest(models, excluded)
    optimizer = _digest(state['optimizer'], {p[1:] for p in excluded if p[0] == 'optimizer'})
    if not model[2] or not optimizer[2] or not optimizer[3] or min(optimizer[3]) < 1:
        raise ValueError('Model and stepped Adam state required; reject fresh initialization')
    if not all(optimizer[4].values()):
        raise ValueError('Missing Adam moments')
    return dict(model_sha256=model[0], optimizer_sha256=optimizer[0],
                model_canonical_sha256=_canonical_digest(models, excluded),
                optimizer_canonical_sha256=_canonical_digest(state['optimizer'],
                    {p[1:] for p in excluded if p[0] == 'optimizer'}),
                model_tensors=model[1], model_elements=model[2],
                optimizer_tensors=optimizer[1], optimizer_elements=optimizer[2],
                optimizer_steps=optimizer[3], moment_elements=optimizer[4])


def compare_state(actual, expected):
    actual_hash, expected_hash = fingerprint(actual), fingerprint(expected)
    if actual_hash != expected_hash:
        raise ValueError('Loaded model/optimizer state differs from checkpoint payload')
    return actual_hash


def _disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return not (a.is_relative_to(b) or b.is_relative_to(a))


def validate_restore_args(args, root, iteration):
    root = Path(root).resolve()
    for flag in ('finetune', 'no_load_optim', 'no_save_optim', 'debug_disable_optimizer', 'async_save'):
        if getattr(args, flag, False):
            raise ValueError('Restore forbids ' + flag)
    if args.start_rollout_id is not None or args.num_rollout != iteration + 2:
        raise ValueError('Infer resume iteration from checkpoint; require exactly one more rollout')
    if getattr(args, 'ckpt_step', None) is not None or args.megatron_to_hf_mode != 'raw':
        raise ValueError('Restore requires tracker-selected native raw checkpoints')
    for role, load, save in [('actor', args.load, args.save), ('critic', args.critic_load, args.critic_save)]:
        expected = root / role
        if load is None or Path(load).resolve() != expected:
            raise ValueError('Unexpected restore load path for ' + role)
        tracker = expected / 'latest_checkpointed_iteration.txt'
        if not tracker.is_file() or tracker.read_text().strip() != str(iteration):
            raise ValueError('Missing or wrong restore tracker for ' + role)
        if not (expected / f'iter_{iteration:07d}').is_dir():
            raise ValueError('Missing selected checkpoint for ' + role)
        if save is None or not _disjoint(save, root):
            raise ValueError('Restore output overlaps original checkpoint')
    if not _disjoint(args.save, args.critic_save):
        raise ValueError('Actor and critic output roots overlap')


def regenerate_state(generate, positional, keywords):
    """Recreate the native state template without changing its load/save mode.

    The native loader's ``is_loading`` mode determines optimizer sharding and
    metadata. Changing it before the independent DCP read can create a valid
    but structurally different optimizer state and produce a false mismatch.
    """
    bound = inspect.signature(generate).bind(*positional, **keywords)
    return generate(*bound.args, **bound.kwargs)


def cpu_template(value, *, preserve=False):
    """Allocate independent CPU targets; never permit a diagnostic read into live state."""
    if isinstance(value, torch.Tensor):
        if preserve:
            return value.detach().to('cpu', copy=True)
        result = torch.empty_like(value, device='cpu')
        result.fill_(float('nan') if result.is_floating_point() else 0)
        return result
    if type(value).__module__.startswith('megatron.core.dist_checkpointing') and hasattr(value, 'data'):
        result = copy.copy(value)
        result.data = cpu_template(value.data, preserve=preserve or _is_nonpersistent(value))
        return result
    if isinstance(value, dict):
        return {k: cpu_template(v, preserve=preserve) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_template(v, preserve=preserve) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_template(v, preserve=preserve) for v in value)
    return copy.deepcopy(value)


def validate_progress(before, after):
    if after['optimizer_steps'] != [step + 1 for step in before['optimizer_steps']]:
        raise ValueError('Expected exactly one additional Adam step')
    if after['optimizer_sha256'] == before['optimizer_sha256']:
        raise ValueError('Optimizer did not change after update')
    for key in ('model_tensors', 'model_elements', 'optimizer_tensors', 'optimizer_elements', 'moment_elements'):
        if before[key] != after[key]:
            raise ValueError('State layout changed across update: ' + key)


def checked_load(load, captured, reader, selected, iteration):
    result = load()
    if not isinstance(result, tuple) or result[0] != iteration:
        raise ValueError('Native loader returned wrong resume iteration')
    captured['native_result'] = result
    actual = regenerate_state(captured['generate'], captured['positional'], captured['keywords'])
    # Keep only model and optimizer. Reading RNG, args or scheduler would add
    # irrelevant state and can trigger GPU allocations when unpickled.
    actual = {k: v for k, v in actual.items() if k == 'optimizer' or k == 'model'
              or k.startswith('model') and k[5:].isdigit()}
    actual_hash = fingerprint(actual)
    # Megatron unwraps LocalNonpersistentObject on load. Exclude precisely those
    # template paths on both sides, never arbitrary keys named "padding".
    excluded = set(_nonpersistent_paths(actual))
    expected = reader(cpu_template(actual), selected, **captured['load_kwargs'])
    expected_hash = fingerprint(expected, excluded=excluded)
    if actual_hash['model_canonical_sha256'] != expected_hash['model_canonical_sha256']:
        raise ValueError('Loaded model state differs from checkpoint payload')
    if actual_hash['optimizer_canonical_sha256'] != expected_hash['optimizer_canonical_sha256']:
        raise ValueError('Loaded model/optimizer state differs from checkpoint payload')
    captured['payload_evidence'] = {
        'model_payload_exact': actual_hash['model_sha256'] == expected_hash['model_sha256'],
        'optimizer_path_hash_exact': actual_hash['optimizer_sha256'] == expected_hash['optimizer_sha256'],
        'optimizer_canonical_exact': True,
    }
    # Prove diagnostics did not change the live optimizer/model either.
    if fingerprint(actual) != actual_hash:
        raise ValueError('Diagnostic checkpoint read modified live training state')
    return actual_hash


def restore_load_checkpoint(args, load, model, optimizer, scheduler, **kwargs):
    """Wrap the actual native load; called on every training rank before any step."""
    import os
    if not os.environ.get('BENCH_RESTORE_ROOT'):
        return load(model, optimizer, scheduler, **kwargs)
    from megatron.training import checkpointing
    from megatron.core import dist_checkpointing
    from benchmark.runtime_evidence import emit

    iteration = int(os.environ['BENCH_RESTORE_ITERATION'])
    role = args.bench_role
    expected_root = Path(os.environ['BENCH_RESTORE_ROOT']).resolve() / role
    if Path(args.load).resolve() != expected_root:
        raise ValueError('Worker load path differs from requested role checkpoint')
    if any(getattr(args, key, False) for key in ('finetune', 'no_load_optim', 'no_save_optim')):
        raise ValueError('Worker selected a fresh-init/optimizer-skip fallback')
    tracker = expected_root / 'latest_checkpointed_iteration.txt'
    if tracker.read_text().strip() != str(iteration):
        raise ValueError('Worker checkpoint tracker changed')
    captured = {}
    generate, reader = checkpointing.generate_state_dict, dist_checkpointing.load

    def capture_generate(*positional, **keywords):
        captured.update(generate=generate, positional=positional, keywords=keywords)
        return generate(*positional, **keywords)

    def capture_read(*positional, **keywords):
        if 'load_kwargs' in captured:
            raise ValueError('Unexpected multiple distributed checkpoint reads')
        bound = inspect.signature(reader).bind(*positional, **keywords)
        # Retain the exact native sharding strategy/strictness/backend options.
        bound.arguments.pop(next(iter(inspect.signature(reader).parameters)), None)
        bound.arguments.pop('checkpoint_dir', None)
        options = dict(bound.arguments)
        for name, parameter in inspect.signature(reader).parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                options.update(options.pop(name, {}))
        captured['load_kwargs'] = options
        return reader(*positional, **keywords)

    checkpointing.generate_state_dict = capture_generate
    dist_checkpointing.load = capture_read
    try:
        snapshot = checked_load(lambda: load(model, optimizer, scheduler, **kwargs),
                                captured, reader, expected_root / f'iter_{iteration:07d}', iteration)
    finally:
        checkpointing.generate_state_dict = generate
        dist_checkpointing.load = reader
    # Preserve native save-template arguments for a second exact read after save.
    _RESTORE_CONTEXT[role] = (captured, snapshot)
    emit(args, 'restore_loaded', loaded_iteration=iteration, next_rollout_id=iteration + 1,
         load_root=str(expected_root), payload_exact=True,
         **captured.get('payload_evidence', {}), snapshot=snapshot)
    return captured['native_result']


def restore_after_save(args, iteration):
    """Verify that the next on-disk checkpoint equals the live post-update state."""
    import os
    if not os.environ.get('BENCH_RESTORE_ROOT'):
        return
    from megatron.core import dist_checkpointing
    from benchmark.runtime_evidence import emit
    if iteration != int(os.environ['BENCH_RESTORE_ITERATION']) + 1:
        raise ValueError('Saved wrong iteration after restore')
    captured, restored = _RESTORE_CONTEXT[args.bench_role]
    snapshot = checked_load(lambda: (iteration, 0), captured, dist_checkpointing.load,
                            Path(args.save) / f'iter_{iteration:07d}', iteration)
    validate_progress(restored, snapshot)
    emit(args, 'restore_saved', saved_iteration=iteration, save_root=str(Path(args.save).resolve()),
         payload_exact=True, **captured.get('payload_evidence', {}), snapshot=snapshot)
