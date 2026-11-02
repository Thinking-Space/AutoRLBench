# SPDX-License-Identifier: Apache-2.0
"""Incident continuation: exact native restore evidence, before training resumes.

Adapted from proposals/miles_ppo_empty_shard_collective/justrl2/restore_evidence.py
in AutoRLBench (Apache-2.0); modified for the step-3 incident scene. Full license:
../runtime_support/LICENSE (retain that license when staging this module).

Pinned Megatron: 37f91617b95c1ef4f01fc66210e502572ad97fda. This adapter observes
native loading; it never restores, repairs, or advances training state itself.

Only trusted local Megatron checkpoints are supported. Tensor hashes include
all bytes (including BF16), never just sampled values. No checkpoint is written
by this module. Imports of Megatron are deferred until running inside a worker.
"""
import copy
import hashlib
import inspect
import json
import math
import os
import random

import numpy as np
from pathlib import Path

import torch

# Process-local only: never attach live models/callables to args, which Megatron
# serializes into common.pt on its next checkpoint save.
_RESTORE_CONTEXT = {}


def _unwrap(value):
    # Megatron's sharded wrappers carry tensors/objects in .data. Avoid importing
    # Megatron for CPU tests or treating a Tensor's .data as another wrapper.
    if _is_nonpersistent(value):
        return value.unwrap()
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


def _storage_padding_paths(value, path=()):
    """Native dp_reshardable drops these explicitly tagged alignment records.

    Identify them only in the live optimizer template, never from file values.
    A plain dictionary key named padding does not authorize any exclusion.
    """
    if isinstance(value, dict):
        marker = value.get('padding')
        if (path and path[0] == 'optimizer' and 'param_state' in path
                and _is_nonpersistent(marker) and marker.unwrap() is True):
            names = set(value) - {'padding'}
            if names != {'param', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unexpected native optimizer padding record')
            if not all(type(value[name]).__module__.startswith('megatron.core.dist_checkpointing')
                       and type(value[name]).__name__ == 'ShardedTensor'
                       and isinstance(value[name].data, torch.Tensor) for name in names):
                raise ValueError('Optimizer padding must carry native ShardedTensor metadata')
            yield path
            return
    value = _unwrap(value)
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    for key, item in items:
        yield from _storage_padding_paths(item, (*path, str(key)))


def _padding_counts(state, paths):
    leaves = dict(_leaves(state))
    tensors = [v for k, v in leaves.items() if isinstance(v, torch.Tensor)
               and any(k[:len(p)] == p for p in paths)]
    return len(tensors), sum(v.numel() for v in tensors)


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



def fingerprint(state, *, excluded=(), storage_padding=None):
    storage_padding = set(_storage_padding_paths(state)) if storage_padding is None else set(storage_padding)
    padding_tensors, padding_elements = _padding_counts(state, storage_padding)
    excluded = set(excluded) | storage_padding
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
                model_tensors=model[1], model_elements=model[2],
                optimizer_tensors=optimizer[1], optimizer_elements=optimizer[2],
                optimizer_steps=optimizer[3], moment_elements=optimizer[4],
                ignored_storage_padding_tensors=padding_tensors,
                ignored_storage_padding_elements=padding_elements)


def state_differences(actual, expected, *, excluded=(), limit=12):
    """Report bounded field-level diagnostics; never change the compared state."""
    a = dict(_leaves(actual, excluded=excluded))
    b = dict(_leaves(expected, excluded=excluded))
    differences = []
    for path in sorted(set(a) | set(b)):
        if len(differences) >= limit:
            break
        row = {'path': list(path)}
        if path not in a or path not in b:
            differences.append({**row, 'kind': 'missing_leaf', 'missing_in': 'actual' if path not in a else 'expected'})
            continue
        x, y = a[path], b[path]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype:
                differences.append({**row, 'kind': 'tensor_metadata',
                    'actual': [str(x.dtype), list(x.shape)], 'expected': [str(y.dtype), list(y.shape)]})
                continue
            count, samples, offset = 0, [], 0
            for xs, ys in zip(x.detach().reshape(-1).split(1024*1024), y.detach().reshape(-1).split(1024*1024)):
                xc, yc = xs.cpu(), ys.cpu()
                unequal = (xc != yc).nonzero().flatten()
                count += unequal.numel()
                for index in unequal[:max(0, 4-len(samples))].tolist():
                    samples.append({'flat_index': offset+index, 'actual': xc[index].item(), 'expected': yc[index].item()})
                offset += xc.numel()
            if count:
                differences.append({**row, 'kind': 'tensor_values', 'different_elements': count, 'samples': samples})
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or type(x) is not type(y) or x != y:
            differences.append({**row, 'kind': 'scalar_or_type',
                'actual': repr(x)[:200], 'expected': repr(y)[:200]})
    return differences


def compare_state(actual, expected):
    actual_hash, expected_hash = fingerprint(actual), fingerprint(expected)
    if actual_hash != expected_hash:
        raise ValueError('Loaded model/optimizer state differs from checkpoint payload')
    return actual_hash


def _disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return not (a.is_relative_to(b) or b.is_relative_to(a))


def regenerate_state(generate, positional, keywords):
    """Inspect populated live buffers using the native save view.

    The pinned DistributedOptimizer.sharded_state_dict calls load_state_dict
    when is_loading=True. Diagnostic snapshots must not invoke that mutation.
    """
    bound = inspect.signature(generate).bind(*positional, **keywords)
    options = dict(bound.arguments.get('optim_sd_kwargs') or {})
    options['is_loading'] = False
    bound.arguments['optim_sd_kwargs'] = options
    return generate(*bound.args, **bound.kwargs)


def cpu_template(value, *, preserve=False):
    """Allocate independent CPU targets; never permit a diagnostic read into live state."""
    if isinstance(value, torch.Tensor):
        if preserve:
            return value.detach().to('cpu', copy=True)
        result = torch.empty_like(value, device='cpu')
        result.fill_(float('nan') if result.is_floating_point() else 0)
        return result
    if _is_nonpersistent(value):
        return type(value)(cpu_template(value.unwrap(), preserve=True))
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


# The scene is intentionally fixed, not a general checkpoint migration API.
SCENE_ITERATION = 3
FINAL_ITERATION = 7
_FORBIDDEN = ('finetune', 'no_load_optim', 'no_save_optim', 'no_load_rng', 'no_save_rng',
              'debug_disable_optimizer', 'debug_train_only', 'debug_rollout_only',
              'async_save', 'override_opt_param_scheduler')
_SCHEDULER_KEYS = {'max_lr', 'min_lr', 'lr_warmup_steps', 'lr_decay_steps',
                   'lr_decay_style', 'num_steps', 'start_wd', 'end_wd',
                   'wd_incr_style', 'wd_incr_steps'}
_RNG_KEYS = {'random_rng_state', 'np_rng_state', 'torch_rng_state',
             'cuda_rng_state', 'rng_tracker_states'}


def _environment():
    root, iteration = os.environ.get('BENCH_RESTORE_ROOT'), os.environ.get('BENCH_RESTORE_ITERATION')
    if root is None and iteration is None:
        return None
    if not root or iteration != str(SCENE_ITERATION):
        raise ValueError('Both BENCH_RESTORE_ROOT and BENCH_RESTORE_ITERATION=3 are required')
    return Path(root).resolve(), SCENE_ITERATION


def _validate_config(args, iteration):
    if type(iteration) is not int or iteration != SCENE_ITERATION:
        raise ValueError('Incident restore requires checkpoint iteration 3')
    for flag in _FORBIDDEN:
        if getattr(args, flag, False):
            raise ValueError('Restore forbids ' + flag)
    if getattr(args, 'pretrained_checkpoint', None) is not None:
        raise ValueError('Restore forbids pretrained_checkpoint fallback')
    if getattr(args, 'start_rollout_id', None) is not None or getattr(args, 'num_rollout', None) != 8:
        raise ValueError('Infer resume from step 3; require exactly rollouts 4..7 (num_rollout=8)')
    if getattr(args, 'ckpt_step', None) is not None or getattr(args, 'megatron_to_hf_mode', None) != 'raw':
        raise ValueError('Restore requires tracker-selected native raw checkpoints')
    if getattr(args, 'ckpt_format', 'torch_dist') != 'torch_dist':
        raise ValueError('Restore requires native torch_dist checkpoints')
    if not getattr(args, 'use_critic', False):
        raise ValueError('Incident restore requires both actor and critic')
    for name, expected in (
        ('tensor_model_parallel_size', 1), ('pipeline_model_parallel_size', 1),
        ('context_parallel_size', 1), ('expert_model_parallel_size', 1),
        ('expert_tensor_parallel_size', 1), ('data_parallel_size', 1),
        ('actor_num_nodes', 1), ('critic_num_nodes', 1),
        ('actor_num_gpus_per_node', 1), ('critic_num_gpus_per_node', 1),
        ('rollout_num_gpus', 1), ('rollout_num_gpus_per_engine', 1), ('num_gpus_per_node', 3),
    ):
        if hasattr(args, name) and getattr(args, name) != expected:
            raise ValueError('Unexpected incident topology: ' + name)


def _selected(root, iteration):
    tracker = root / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file() or tracker.read_text().strip() != str(iteration):
        raise ValueError('Missing or wrong restore tracker: ' + str(root))
    selected = root / f'iter_{iteration:07d}'
    if not selected.is_dir():
        raise ValueError('Missing selected checkpoint: ' + str(selected))
    return selected


def validate_restore_args(args, root, iteration):
    """Driver gate, before role-specific worker args.load/args.save rebinding."""
    _validate_config(args, iteration)
    root = Path(root).resolve()
    for role, load, save in [('actor', args.load, args.save),
                             ('critic', args.critic_load, args.critic_save)]:
        expected = root / role
        if load is None or Path(load).resolve() != expected:
            raise ValueError('Unexpected restore load path for ' + role)
        _selected(expected, iteration)
        if save is None or not _disjoint(save, root):
            raise ValueError('Restore output overlaps original checkpoint')
    if not _disjoint(args.save, args.critic_save):
        raise ValueError('Actor and critic output roots overlap')


def _aux_hash(value):
    """Full typed bytes, including NumPy RNG arrays and empty containers."""
    sha = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Generator):
            item = item.get_state()
        if isinstance(item, torch.Tensor):
            sha.update(b'tensor:' + _digest({'value': item})[0].encode())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('Unsupported object array in native state')
            sha.update(json.dumps(['ndarray', item.dtype.str, list(item.shape)]).encode())
            sha.update(item.tobytes(order='C'))
        elif isinstance(item, dict):
            sha.update(b'dict:' + str(len(item)).encode() + b':')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            sha.update(type(item).__name__.encode() + str(len(item)).encode() + b':')
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            sha.update(type(item).__name__.encode() + b':' + json.dumps(item, allow_nan=False).encode() + b'\0')
        elif isinstance(item, bytes):
            sha.update(b'bytes:' + str(len(item)).encode() + b':' + item)
        else:
            raise ValueError('Unrecognized native state schema: ' + str(type(item)))
    visit(value)
    return sha.hexdigest()


def _rng_snapshot(value):
    if not isinstance(value, dict) or set(value) != _RNG_KEYS:
        raise ValueError('RNG schema unavailable/unrecognized; cannot verify restore')
    result = copy.deepcopy({k: v for k, v in value.items() if k != 'rng_tracker_states'})
    tracker = value['rng_tracker_states']
    if not isinstance(tracker, dict) or not tracker:
        raise ValueError('Missing native RNG tracker state')
    result['rng_tracker_states'] = {}
    for name, state in tracker.items():
        if isinstance(state, torch.Generator):
            state = state.get_state()
        result['rng_tracker_states'][name] = state
    for state in [result['torch_rng_state'], result['cuda_rng_state'],
                  *result['rng_tracker_states'].values()]:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.numel() == 0:
            raise ValueError('Unrecognized native RNG byte state')
    if not isinstance(result['random_rng_state'], tuple) or len(result['random_rng_state']) != 3:
        raise ValueError('Unrecognized Python RNG schema')
    if not isinstance(result['np_rng_state'], tuple) or len(result['np_rng_state']) != 5:
        raise ValueError('Unrecognized NumPy RNG schema')
    if not isinstance(result['np_rng_state'][1], np.ndarray):
        raise ValueError('Missing NumPy RNG array')
    return cpu_template(result, preserve=True)


def capture_auxiliary(scheduler, tensor_parallel):
    """Read live state immediately after native load; never reseed or step it."""
    if scheduler is None:
        raise ValueError('Native scheduler is missing')
    rng = dict(random_rng_state=random.getstate(), np_rng_state=np.random.get_state(),
               torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
               rng_tracker_states=tensor_parallel.get_cuda_rng_tracker().get_states())
    return dict(scheduler=cpu_template(scheduler.state_dict(), preserve=True), rng=_rng_snapshot(rng))


def compare_auxiliary(actual, common, payload, iteration):
    """Common scheduler plus independently read native RNG ShardedObject."""
    if not isinstance(common, dict) or type(common.get('iteration')) is not int or common['iteration'] != iteration:
        raise ValueError('common.pt iteration unavailable or differs from native load')
    expected_scheduler = common.get('opt_param_scheduler')
    live_scheduler = actual['scheduler']
    for state in (expected_scheduler, live_scheduler):
        if not isinstance(state, dict) or not _SCHEDULER_KEYS <= state.keys():
            raise ValueError('common.pt scheduler schema unavailable/unrecognized; cannot verify restore')
        if isinstance(state['num_steps'], bool) or not isinstance(state['num_steps'], (int, float)) or state['num_steps'] <= 0:
            raise ValueError('Missing restored scheduler progress')
    scheduler_hash = _aux_hash(live_scheduler)
    if scheduler_hash != _aux_hash(expected_scheduler):
        raise ValueError('Live scheduler differs from common.pt checkpoint payload')
    rng = payload.get('rng_state') if isinstance(payload, dict) else None
    if not isinstance(rng, list) or len(rng) != 1:
        raise ValueError('DCP RNG schema unavailable/unrecognized; expected one TP1/CP1/DP1 state')
    rng_hash = _aux_hash(_rng_snapshot(actual['rng']))
    if rng_hash != _aux_hash(_rng_snapshot(rng[0])):
        raise ValueError('Live RNG differs from independent DCP checkpoint payload')
    # Some formats keep RNG in common state too; never silently ignore a conflict.
    if 'rng_state' in common and _aux_hash(common['rng_state']) != _aux_hash(rng):
        raise ValueError('common.pt RNG differs from independent DCP checkpoint payload')
    return dict(scheduler_exact=True, rng_exact=True, scheduler_sha256=scheduler_hash,
                rng_sha256=rng_hash, scheduler_num_steps=live_scheduler['num_steps'],
                scheduler_source='common.pt:opt_param_scheduler', rng_source='DCP:rng_state')


def validate_progress(before, after, *, step_delta=1):
    if type(step_delta) is not int or not 1 <= step_delta <= FINAL_ITERATION - SCENE_ITERATION:
        raise ValueError('Expected between one and four additional Adam steps')
    if after['optimizer_steps'] != [step + step_delta for step in before['optimizer_steps']]:
        raise ValueError('Expected exactly ' + str(step_delta) + ' additional Adam steps')
    if after['optimizer_sha256'] == before['optimizer_sha256']:
        raise ValueError('Optimizer did not change after update')
    for key in ('model_tensors', 'model_elements', 'optimizer_tensors', 'optimizer_elements', 'moment_elements',
                'ignored_storage_padding_tensors', 'ignored_storage_padding_elements'):
        if before[key] != after[key]:
            raise ValueError('State layout changed across update: ' + key)


def checked_load(load, captured, reader, selected, iteration, *, capture_aux=None):
    """Read back into independent CPU targets, without touching live state."""
    result = load()
    # This must precede regenerating templates or any other diagnostic I/O.
    auxiliary = capture_aux() if capture_aux is not None else None
    if not isinstance(result, tuple) or len(result) != 2 or type(result[0]) is not int or result[0] != iteration:
        raise ValueError('Native loader returned wrong resume iteration')
    if not {'generate', 'positional', 'keywords', 'load_kwargs'} <= captured.keys():
        raise ValueError('Native loader did not generate and read a distributed checkpoint')
    captured['native_result'] = result
    regenerated = regenerate_state(captured['generate'], captured['positional'], captured['keywords'])
    actual = {k: v for k, v in regenerated.items() if k == 'optimizer' or k == 'model'
              or k.startswith('model') and k[5:].isdigit()}
    actual_hash = fingerprint(actual)
    storage_padding = set(_storage_padding_paths(actual))
    excluded = set(_nonpersistent_paths(actual)) | storage_padding
    template = dict(actual)
    if auxiliary is not None:
        if 'rng_state' not in regenerated or regenerated['rng_state'] is None:
            raise ValueError('Native loader omitted RNG restore template')
        template['rng_state'] = regenerated['rng_state']
    expected = reader(cpu_template(template), str(selected), **captured['load_kwargs'])
    if not isinstance(expected, dict):
        raise ValueError('Unsupported native DCP return schema')
    expected_hash = fingerprint(expected, excluded=excluded, storage_padding=storage_padding)
    if actual_hash != expected_hash:
        detail = {'actual': actual_hash, 'expected': expected_hash,
                  'differences': state_differences(actual, {k: v for k, v in expected.items() if k in actual}, excluded=excluded)}
        print('RESTORE_MISMATCH=' + json.dumps(detail, sort_keys=True), flush=True)
        raise ValueError('Loaded model/optimizer state differs from checkpoint payload: '
                         + json.dumps(detail['differences'], sort_keys=True))
    if auxiliary is not None:
        try:
            # Trusted bench-owned checkpoint only, matching the pinned common strategy.
            common = torch.load(Path(selected) / 'common.pt', map_location='cpu', weights_only=False)
        except Exception as exc:
            raise ValueError('common.pt schema unavailable; cannot verify scheduler/RNG restore') from exc
        captured['auxiliary_evidence'] = compare_auxiliary(auxiliary, common, expected, iteration)
    final_state = regenerate_state(captured['generate'], captured['positional'], captured['keywords'])
    if fingerprint(final_state) != actual_hash:
        raise ValueError('Diagnostic checkpoint read modified live training state')
    if auxiliary is not None and _aux_hash(capture_aux()) != _aux_hash(auxiliary):
        raise ValueError('Diagnostic checkpoint read modified live scheduler/RNG state')
    return actual_hash


def restore_load_checkpoint(args, load, model, optimizer, scheduler, **kwargs):
    """Install around native load(model, optimizer, scheduler, **kwargs)."""
    env = _environment()
    if env is None:
        return load(model, optimizer, scheduler, **kwargs)
    root, iteration = env
    role = getattr(args, 'bench_role', None)
    if role not in ('actor', 'critic'):
        raise ValueError('args.bench_role must identify actor or critic before restore')
    _RESTORE_CONTEXT.pop(role, None)
    _validate_config(args, iteration)
    expected_root = root / role
    if getattr(args, 'load', None) is None or Path(args.load).resolve() != expected_root:
        raise ValueError('Worker load path differs from requested role checkpoint')
    if getattr(args, 'save', None) is None or not _disjoint(args.save, root):
        raise ValueError('Worker restore output overlaps original checkpoint')
    selected = _selected(expected_root, iteration)
    if kwargs.get('skip_load_to_model_and_opt', False) or kwargs.get('load_arg', 'load') != 'load':
        raise ValueError('Worker selected a native-load fallback or skipped applying state')
    if optimizer is None or scheduler is None:
        raise ValueError('A live optimizer and scheduler are required')
    from megatron.training import checkpointing
    from megatron.core import dist_checkpointing, mpu, tensor_parallel
    from benchmark.runtime_evidence import emit

    for name in ('get_tensor_model_parallel_world_size', 'get_pipeline_model_parallel_world_size',
                 'get_context_parallel_world_size', 'get_data_parallel_world_size',
                 'get_expert_model_parallel_world_size'):
        if getattr(mpu, name)() != 1:
            raise ValueError('Actual native worker topology differs: ' + name)
    captured = {}
    generate, reader = checkpointing.generate_state_dict, dist_checkpointing.load

    def capture_generate(*positional, **keywords):
        captured.update(generate=generate, positional=positional, keywords=keywords)
        return generate(*positional, **keywords)

    def capture_read(*positional, **keywords):
        if 'load_kwargs' in captured:
            raise ValueError('Unexpected multiple distributed checkpoint reads')
        signature = inspect.signature(reader)
        bound = signature.bind(*positional, **keywords)
        if Path(bound.arguments.get('checkpoint_dir', '')).resolve() != selected.resolve():
            raise ValueError('Native DCP read selected a different checkpoint')
        bound.arguments.pop(next(iter(signature.parameters)), None)
        bound.arguments.pop('checkpoint_dir', None)
        options = dict(bound.arguments)
        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                options.update(options.pop(name, {}))
        captured['load_kwargs'] = options
        return reader(*positional, **keywords)

    checkpointing.generate_state_dict = capture_generate
    dist_checkpointing.load = capture_read
    try:
        snapshot = checked_load(lambda: load(model, optimizer, scheduler, **kwargs),
            captured, reader, selected, iteration,
            capture_aux=lambda: capture_auxiliary(scheduler, tensor_parallel))
        if any(step != 4 for step in snapshot['optimizer_steps']):
            raise ValueError('Step-3 incident checkpoint must restore Adam counter 4')
        _selected(expected_root, iteration)
        _validate_config(args, iteration)
    finally:
        checkpointing.generate_state_dict = generate
        dist_checkpointing.load = reader
    # No model, optimizer, callback, or scheduler is ever attached to serialized args.
    emit(args, 'restore_loaded', loaded_iteration=iteration, next_rollout_id=iteration + 1,
         load_root=str(expected_root), payload_exact=True, snapshot=snapshot,
         **captured['auxiliary_evidence'])
    _RESTORE_CONTEXT[role] = dict(captured=captured, restored=snapshot, optimizer=optimizer,
        root=root, save=Path(args.save).resolve(), last_rollout=iteration)
    return captured['native_result']


def restore_after_optimizer_step(args, optimizer, rollout_id, step_id=0):
    """Optional evidence hook after each successful native update (one per rollout).

    Call for actor and critic after the real optimizer step; this never performs
    an update. Existing optimizer_step evidence still supplies success/grad norms.
    """
    env = _environment()
    if env is None:
        return
    from benchmark.runtime_evidence import emit
    context = _RESTORE_CONTEXT.get(args.bench_role)
    if context is None or context['optimizer'] is not optimizer or context['root'] != env[0]:
        raise ValueError('Missing matching native restore context')
    if type(rollout_id) is not int or rollout_id != context['last_rollout'] + 1 or not 4 <= rollout_id <= 7 or step_id != 0:
        raise ValueError('Expected one ordered native optimizer update for each rollout 4..7')
    steps = []
    for path, value in _leaves(optimizer.state_dict()):
        if path and path[-1] == 'step':
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != rollout_id + 1:
                raise ValueError('Live Adam counter differs from resumed rollout')
            steps.append(float(value))
    if not steps:
        raise ValueError('Live Adam counters are missing')
    emit(args, 'restore_optimizer_step', rollout_id=rollout_id, step_id=step_id,
         optimizer_steps=sorted(steps), expected_optimizer_step=rollout_id + 1)
    context['last_rollout'] = rollout_id


def restore_after_save(args, iteration):
    """Optional synchronous post-save readback for any resumed iteration 4..7."""
    env = _environment()
    if env is None:
        return
    from megatron.core import dist_checkpointing
    from benchmark.runtime_evidence import emit
    context = _RESTORE_CONTEXT.get(args.bench_role)
    if context is None or context['root'] != env[0] or Path(args.save).resolve() != context['save']:
        raise ValueError('Missing matching native restore/save context')
    if type(iteration) is not int or not 4 <= iteration <= FINAL_ITERATION:
        raise ValueError('Saved iteration must be within incident rollouts 4..7')
    if not _disjoint(args.save, env[0]):
        raise ValueError('Restore output overlaps original checkpoint')
    snapshot = checked_load(lambda: (iteration, 0), context['captured'], dist_checkpointing.load,
                            context['save'] / f'iter_{iteration:07d}', iteration)
    delta = iteration - SCENE_ITERATION
    validate_progress(context['restored'], snapshot, step_delta=delta)
    emit(args, 'restore_saved', saved_iteration=iteration, step_delta=delta,
         save_root=str(context['save']), payload_exact=True, snapshot=snapshot)
