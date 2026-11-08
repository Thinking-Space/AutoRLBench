#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute native PPO on CPU/Gloo (Python >=3.10; source may require >=3.12).

run_replay(source, output, *, cp=2, dp=2, order=None, steps=(11, 3),
           seed=1729, mask_mode="ones", replay=None) -> aggregate result.
--replay uses one authored sample per DP replica for one update, overriding
synthetic --steps/--mask-mode. Source code is executable and is not sandboxed.
Only config.json, result.json, and rankN.stdout.log survive a run.
Exit codes: 0 complete,
1 native execution failed, 2 infrastructure/arguments/source changed.
"""
from __future__ import annotations
import argparse
import ast
from datetime import timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

DEFAULT_SEED = 1729
COLLECTIVE_TIMEOUT = 2.0
OUTER_TIMEOUT = 45.0
MAX_RESPONSE_LENGTH = 4096
MAX_STEPS = 16
PLAN = "miles/utils/collective_plan.py"
NATIVE_FUNCTIONS = {
    "miles/backends/training_utils/cp_utils.py": (
        "get_logits_and_tokens_offset_with_cp", "local_response_to_full", "slice_log_prob_with_cp"),
    "miles/utils/ppo_utils.py": (
        "length_adaptive_lambda", "get_advantages_and_returns_batch", "chunked_gae", "vanilla_gae"),
    "miles/utils/distributed_utils.py": ("distributed_masked_whiten",),
    "miles/backends/training_utils/loss.py": ("compute_advantages_and_returns",),
}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _integers(value, name):
    if isinstance(value, str):
        try:
            value = [int(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{name} must be comma-separated integers") from exc
    try:
        value = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if not value or any(type(x) is not int for x in value):
        raise ValueError(f"{name} must contain integers")
    return value


def _configuration(cp, dp, order, steps, mask_mode, seed):
    if type(cp) is not int or cp not in (1, 2, 4):
        raise ValueError("cp must be 1, 2, or 4")
    if type(dp) is not int or dp not in (2, 3) or cp * dp > 12:
        raise ValueError("dp must be 2 or 3, with world size <= 12")
    steps = _integers(steps, "steps")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"at most {MAX_STEPS} updates are supported")
    if any(not 1 <= length <= MAX_RESPONSE_LENGTH for length in steps):
        raise ValueError(f"response lengths must be between 1 and {MAX_RESPONSE_LENGTH}")
    order = list(range(cp * dp)) if order is None else _integers(order, "order")
    if sorted(order) != list(range(cp * dp)):
        raise ValueError("order must be a permutation of every global rank")
    rows = [order[d * cp:(d + 1) * cp] for d in range(dp)]
    columns = [order[c::cp] for c in range(cp)]
    if any(group != sorted(group) for group in rows + columns):
        raise ValueError("order rows and columns must be sorted to preserve native subgroup ranks")
    if mask_mode not in ("ones", "sparse"):
        raise ValueError("mask_mode must be ones or sparse")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return dict(cp=cp, dp=dp, order=order,
                steps=steps, mask_mode=mask_mode, seed=seed)


def _samples(length, cp, dp_rank, seed, step, mask_mode):
    """Deterministic inputs, with complementary prompts across replicas."""
    rng = random.Random(f"{seed}:{step}:{dp_rank}")
    samples = []
    for sample in range(1):
        if dp_rank == 0:
            total = math.ceil((length + 32 + sample * 7) / (2 * cp)) * (2 * cp)
        else:
            total = max(4 * cp, math.ceil((length + 1) / (2 * cp)) * (2 * cp)) + (dp_rank - 1) * 2 * cp
        values = [0.17 * dp_rank + 0.09 * sample + 0.025 * index + rng.uniform(-0.3, 0.3)
                  for index in range(length)]
        masks = [float(not (mask_mode == "sparse" and length > 3
                            and (index + dp_rank + seed) % 4 == 0)) for index in range(length)]
        if mask_mode == "sparse":
            # Keep at least one valid token in each nonempty response shard.
            # This workload exercises sparse masks, not globally absent targets.
            width = total // (2 * cp)
            represented = set()
            for index in range(length):
                block = (total - length - 1 + index) // width
                coordinate = min(block, 2 * cp - 1 - block) if cp > 1 else 0
                if coordinate not in represented:
                    masks[index] = 1.0
                    represented.add(coordinate)
        samples.append(dict(total=total, length=length, values=values, masks=masks,
                            reward=rng.uniform(-1.0, 1.0) + 0.4 * dp_rank + 0.13 * sample))
    return samples


def _fixture_samples(path, dp):
    rows = json.loads(Path(path).read_text())['samples']
    if not isinstance(rows, list) or len(rows) != dp:
        raise ValueError('replay must contain exactly one sample per DP replica')
    samples = []
    for row in rows:
        length, total = row['response_length'], len(row['tokens'])
        mask, reward = row['loss_mask'], row['reward']
        if (type(length) is not int or not 1 <= length <= MAX_RESPONSE_LENGTH
                or not length < total <= MAX_RESPONSE_LENGTH
                or not isinstance(mask, list) or len(mask) != length
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in mask)
                or type(reward) not in (int, float) or not math.isfinite(reward)):
            raise ValueError('invalid replay sample')
        samples.append(dict(total=total, length=length, masks=mask, reward=reward,
                            values=[0.52] * length))
    return samples


def _bound_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)
                for a in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


def _source_functions(source):
    """Select unchanged native definitions and their local helper dependencies."""
    selected, hashes = {}, {}
    plan_needed = False
    provided = {'torch', 'dist', 'F', 'get_parallel_state', 'Namespace', 'RolloutBatch'}
    provided.update(name for names in NATIVE_FUNCTIONS.values() for name in names)
    native_modules = {p[:-3].replace('/', '.') for p in NATIVE_FUNCTIONS}
    for relative, required in NATIVE_FUNCTIONS.items():
        path = Path(source) / relative
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        tree = ast.parse(data, filename=str(path))
        definitions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if not set(required) <= definitions:
            raise ValueError('missing native functions')
        by_name = {name: node for node in tree.body for name in _bound_names(node)}
        chosen = {by_name[name] for name in required}
        while True:
            needed = {n.id for node in chosen for n in ast.walk(node)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            extra = {by_name[name] for name in needed if name in by_name
                     and (name not in provided or isinstance(by_name[name], ast.FunctionDef))
                     and not (isinstance(by_name[name], ast.ImportFrom)
                              and by_name[name].module in native_modules)}
            if extra <= chosen:
                break
            chosen |= extra
        nodes = [n for n in tree.body if n in chosen]
        # Preserve only imports used by the selected code; Python imports the
        # complete local module normally, including its persistent globals.
        for node in nodes:
            for n in ast.walk(node):
                if isinstance(n, ast.ImportFrom) and (n.module == 'miles.utils.collective_plan'
                        or n.module == 'miles.utils' and any(a.name == 'collective_plan' for a in n.names)):
                    plan_needed = True
                elif isinstance(n, ast.Import) and any(a.name == 'miles.utils.collective_plan' for a in n.names):
                    plan_needed = True
        selected[relative] = nodes
    if plan_needed:
        hashes[PLAN] = hashlib.sha256((Path(source) / PLAN).read_bytes()).hexdigest()
    return selected, hashes


def _load_native(source, namespace):
    selected, _ = _source_functions(source)
    # These packages expose the pinned local tree without importing GPU-only
    # package initializers. No native collective or numerical function is wrapped.
    for name in list(sys.modules):
        if name == 'miles' or name.startswith('miles.'):
            del sys.modules[name]
    for name in ('miles', 'miles.utils', 'miles.backends', 'miles.backends.training_utils'):
        package = ModuleType(name)
        package.__path__ = [str(Path(source).joinpath(*name.split('.')))]
        sys.modules[name] = package
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, package)
    parallel = ModuleType('miles.backends.training_utils.parallel')
    parallel.get_parallel_state = namespace['get_parallel_state']
    sys.modules[parallel.__name__] = parallel
    for relative, nodes in selected.items():
        name = relative[:-3].replace('/', '.')
        module = ModuleType(name)
        module.__dict__.update(namespace)
        module.__name__, module.__file__ = name, str(Path(source) / relative)
        module.__package__ = name.rpartition('.')[0]
        sys.modules[name] = module
        setattr(sys.modules[module.__package__], name.rpartition('.')[2], module)
        body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                     module.__file__, 'exec'), module.__dict__)
        namespace.update({key: value for key, value in module.__dict__.items()
                          if not key.startswith('__')})
    return namespace


def _worker(config, rank, init_method):
    code, dist = 2, None
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        torch.set_num_threads(1)
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
        world = config['cp'] * config['dp']
        order = config['order']
        dp_rank, cp_rank = divmod(order.index(rank), config['cp'])
        dist.init_process_group('gloo', init_method=init_method, rank=rank,
                                world_size=world, timeout=timedelta(seconds=15))
        control = dist.new_group(list(range(world)), timeout=timedelta(seconds=30))
        cp_group = dp_group = None
        for d in range(config['dp']):
            members = order[d * config['cp']:(d + 1) * config['cp']]
            group = dist.new_group(members, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
            if rank in members:
                cp_group = group
        for c in range(config['cp']):
            members = order[c::config['cp']]
            group = dist.new_group(members, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
            if rank in members:
                dp_group = group
        if dist.get_rank(cp_group) != cp_rank or dist.get_rank(dp_group) != dp_rank:
            raise RuntimeError('invalid subgroup rank')
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=config['cp'], group=cp_group),
                                intra_dp=SimpleNamespace(rank=dp_rank, size=config['dp'], group=dp_group),
                                intra_dp_cp=SimpleNamespace(group=dist.group.WORLD))
        namespace = dict(torch=torch, dist=dist, F=F, get_parallel_state=lambda: state,
                         Namespace=SimpleNamespace, RolloutBatch=dict)
        ready = 0
        try:
            _load_native(Path(config['source']), namespace)
            fixtures = _fixture_samples(config['replay'], config['dp']) if config['replay'] else None
        except BaseException:
            ready = 2
        readiness = torch.tensor([ready], dtype=torch.int32)
        dist.all_reduce(readiness, op=dist.ReduceOp.MAX, group=control)
        if readiness.item():
            return 2
        settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator='ppo',
                                   gamma=1.0, lambd=1.0, gae_lambda_k=0.5,
                                   normalize_advantages=True, qkv_format='thd')
        code = 0
        for update, length in enumerate([None] if fixtures is not None else config['steps']):
            failed = 0
            try:
                samples = [fixtures[dp_rank]] if fixtures is not None else _samples(
                    length, config['cp'], dp_rank, config['seed'], update, config['mask_mode'])
                values = [namespace['slice_log_prob_with_cp'](
                    torch.tensor(s['values'], dtype=torch.float32), s['total'], s['length']) for s in samples]
                batch = dict(values=values, log_probs=[torch.zeros_like(v) for v in values],
                             rewards=[s['reward'] for s in samples], total_lengths=[s['total'] for s in samples],
                             response_lengths=[s['length'] for s in samples],
                             loss_masks=[torch.tensor(s['masks'], dtype=torch.float32) for s in samples])
                namespace['compute_advantages_and_returns'](settings, batch)
            except BaseException:
                failed = 1
            # Peers stay alive on a separate group until every native call has
            # completed or reached its own data-group timeout.
            failure = torch.tensor([failed], dtype=torch.int32)
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=control)
            if failure.item():
                code = 1
                break
    except BaseException:
        code = 2
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                code = 2
    return code


def run_replay(source, output, *, cp=2, dp=2, order=None, steps=(11, 3),
               mask_mode='ones', seed=DEFAULT_SEED, replay=None):
    config = _configuration(cp, dp, order, steps, mask_mode, seed)
    source, output = Path(source).resolve(), Path(output).absolute()
    config.update(source=str(source), output=str(output), replay=str(Path(replay).resolve()) if replay else None)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='infrastructure_error', exit_code=2, source_sha256={})
    processes, logs = [], []
    try:
        _write(output / 'config.json', config)
        _, before = _source_functions(source)
        result['source_sha256'] = before
        if replay:
            _fixture_samples(replay, dp)
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           TORCH_DISTRIBUTED_DEBUG='OFF', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        with tempfile.TemporaryDirectory(prefix='.run-', dir=output) as scratch:
            init_method = (Path(scratch) / 'rendezvous').as_uri()
            deadline = time.monotonic() + OUTER_TIMEOUT
            try:
                for rank in range(cp * dp):
                    log = (output / f'rank{rank}.stdout.log').open('w')
                    logs.append(log)
                    processes.append(subprocess.Popen(
                        [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--_worker', str(rank),
                         '--_config', str(output / 'config.json'), '--_init-method', init_method],
                        stdout=log, stderr=subprocess.STDOUT, env=environment))
                while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                    time.sleep(0.05)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for log in logs:
                    log.close()
            _, after = _source_functions(source)
            codes = [p.returncode for p in processes]
            if before == after and len(codes) == cp * dp and all(c in (0, 1) for c in codes):
                failed = any(codes)
                result.update(status='failed' if failed else 'completed', exit_code=int(failed))
    except BaseException:
        pass
    _write(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, help='fresh output directory')
    parser.add_argument('--cp', type=int, default=2)
    parser.add_argument('--dp', type=int, default=2)
    parser.add_argument('--order', help='logical-to-global permutation with sorted rows and columns')
    parser.add_argument('--steps', default='11,3', help='up to 16 response lengths, each 1..4096')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--mask-mode', choices=('ones', 'sparse'), default='ones')
    parser.add_argument('--replay', type=Path, help='authored JSON fixture; one sample per DP replica')
    parser.add_argument('--_worker', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_init-method', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        try:
            return _worker(json.loads(args._config.read_text()), args._worker, args._init_method)
        except BaseException:
            return 2
    if args.output is None:
        parser.error('--output is required')
    try:
        result = run_replay(args.source, args.output, cp=args.cp, dp=args.dp, order=args.order,
                            steps=args.steps, seed=args.seed, mask_mode=args.mask_mode, replay=args.replay)
    except (OSError, ValueError, TypeError):
        result = dict(status='infrastructure_error', exit_code=2, source_sha256={})
    print(json.dumps(result))
    return result['exit_code']


if __name__ == '__main__':
    try:
        exit_code = main()
    except Exception:
        exit_code = 2
    raise SystemExit(exit_code)
