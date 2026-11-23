#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay candidate native PPO functions on CPU/Gloo, with optional route tracing.

Only Python >=3.10 and CPU PyTorch are required (some source revisions require
Python >=3.12 to parse). This executes source code; it is not a sandbox.
No model, optimizer, numerical reference, or correctness verdict is included.

API: run_probe(source, output, *, cp=2, dp=2, order=None, steps=(11, 3),
               mask_mode="ones", seed=1729, trace=False) -> summary dict.
Outputs: summary.json, config.json, rankN.json, rankN.log; with --trace also
rankN.trace.jsonl. Each rank row contains steps with samples
{total,length,values,masks,reward}, sliced_values, raw_advantages, raw_returns,
whitening {values,masks,output}, and final advantages/returns when available.
An exception retains all inputs and partial results already obtained.
Exit codes: 0 completed, 1 candidate error, 2 infrastructure/argument error.
"""
from __future__ import annotations

import argparse
import ast
from datetime import timedelta
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
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
        "_parallel_rank", "_parallel_size", "get_logits_and_tokens_offset_with_cp",
        "local_response_to_full", "slice_log_prob_with_cp"),
    "miles/utils/ppo_utils.py": (
        "length_adaptive_lambda", "get_advantages_and_returns_batch", "chunked_gae", "vanilla_gae"),
    "miles/utils/distributed_utils.py": ("distributed_masked_whiten",),
    "miles/backends/training_utils/loss.py": ("compute_advantages_and_returns",),
}


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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


def _configuration(cp, dp, order, steps, mask_mode, seed, trace):
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
    if type(seed) is not int or type(trace) is not bool:
        raise ValueError("seed must be an integer and trace a boolean")
    return dict(cp=cp, dp=dp, world_size=cp * dp, logical_to_global=order,
                steps=steps, mask_mode=mask_mode, seed=seed, trace=trace,
                collective_timeout=COLLECTIVE_TIMEOUT, outer_timeout=OUTER_TIMEOUT)


def _source_functions(source):
    """Read candidate definitions plus their same-module helper dependencies."""
    selected, hashes = {}, {}
    for relative, required in NATIVE_FUNCTIONS.items():
        path = source / relative
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        tree = ast.parse(data, filename=str(path))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        missing = set(required).difference(functions)
        if missing:
            raise ValueError(f"{relative}: missing native functions {sorted(missing)}")
        names = set(required)
        while True:
            dependencies = {n.id for name in names for n in ast.walk(functions[name])
                            if isinstance(n, ast.Name)}.intersection(functions)
            if dependencies.issubset(names):
                break
            names.update(dependencies)
        # Retain the candidate's actual reduction import, including aliases.
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                   and n.module == "miles.utils.collective_plan"]
        selected[relative] = imports + [n for n in tree.body
                                       if isinstance(n, ast.FunctionDef) and n.name in names]
    data = (source / PLAN).read_bytes()
    ast.parse(data, filename=str(source / PLAN))
    hashes[PLAN] = hashlib.sha256(data).hexdigest()
    return selected, hashes


def _load_native(source, namespace, trace_recorder=None):
    selected, _ = _source_functions(source)
    for name in ("miles", "miles.utils", "miles.backends", "miles.backends.training_utils"):
        package = ModuleType(name)
        package.__path__ = [str(source.joinpath(*name.split(".")))]
        sys.modules[name] = package
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, package)
    name = "miles.utils.collective_plan"
    spec = importlib.util.spec_from_file_location(name, source / PLAN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules["miles.utils"].collective_plan = module
    if trace_recorder is not None:
        # Install before import so candidate `from ... import all_reduce` aliases
        # remain observable. Each worker owns its own Python process.
        trace_recorder.dist.all_reduce = trace_recorder.actual_reduce
    spec.loader.exec_module(module)
    if not callable(getattr(module, "reduce_sum", None)) or not callable(getattr(module, "clear", None)):
        raise ValueError("collective_plan must expose callable reduce_sum and clear")
    # Whole-module execution above preserves imports, globals, and cache lifetime.
    # A fresh worker starts cold; do not clear the cache between updates.
    if trace_recorder is not None:
        module.reduce_sum = trace_recorder.wrap_reduce(module.reduce_sum)
    namespace["reduce_sum"] = module.reduce_sum
    for relative, nodes in selected.items():
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes]
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                       str(source / relative), "exec")
        exec(code, namespace)
        if relative.endswith("cp_utils.py"):
            cp_module = ModuleType("miles.backends.training_utils.cp_utils")
            cp_module.__dict__.update(namespace)
            sys.modules[cp_module.__name__] = cp_module
            sys.modules["miles.backends.training_utils"].cp_utils = cp_module
    return module


class _Trace:
    """Transparent observations at the requested and actual reduction boundaries."""
    def __init__(self, path, rank, dist, context):
        self.stream = path.open("w", buffering=1)
        self.rank, self.dist, self.context = rank, dist, context
        self.native_all_reduce = dist.all_reduce
        self.sequence = 0
        self.call = 0
        self.request = None

    def members(self, group):
        return self.dist.get_process_group_ranks(group if group is not None else self.dist.group.WORLD)

    def emit(self, operation, event, call, tensor, requested, actual, caller, **fields):
        row = dict(sequence=self.sequence, rank=self.rank, step=self.context["step"],
                   phase=self.context["phase"], caller=caller, operation=operation,
                   event=event, call=call, numel=tensor.numel(), shape=list(tensor.shape),
                   requested_members=requested, actual_members=actual, **fields)
        self.sequence += 1
        self.stream.write(json.dumps(_json_value(row), allow_nan=False) + "\n")

    def invoke(self, operation, function, tensor, args, kwargs, requested, actual, caller, call):
        start = time.monotonic()
        self.emit(operation, "enter", call, tensor, requested, actual, caller)
        try:
            result = function(tensor, *args, **kwargs)
        except Exception as exc:
            self.emit(operation, "error", call, tensor, requested, actual, caller,
                      elapsed=time.monotonic() - start, error_type=type(exc).__name__, error=str(exc))
            raise
        self.emit(operation, "return", call, tensor, requested, actual, caller,
                  elapsed=time.monotonic() - start)
        return result

    def wrap_reduce(self, function):
        def reduce_sum(tensor, *args, **kwargs):
            group = kwargs.get("group", args[0] if args else None)
            request = dict(members=self.members(group), caller=sys._getframe(1).f_code.co_name, call=self.call)
            self.call += 1
            previous, self.request = self.request, request
            try:
                return self.invoke("reduce_sum", function, tensor, args, kwargs,
                                   request["members"], None, request["caller"], request["call"])
            finally:
                self.request = previous
        return reduce_sum

    def actual_reduce(self, tensor, *args, **kwargs):
        request = self.request
        if request is None:
            return self.native_all_reduce(tensor, *args, **kwargs)
        group = kwargs.get("group", args[1] if len(args) > 1 else None)
        return self.invoke("all_reduce", self.native_all_reduce, tensor, args, kwargs,
                           request["members"], self.members(group), request["caller"], request["call"])


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


def _worker(config, rank, init_method):
    output, source = Path(config["output"]), Path(config["source"])
    logical = config["logical_to_global"].index(rank)
    dp_rank, cp_rank = divmod(logical, config["cp"])
    record = dict(rank=rank, cp_rank=cp_rank, dp_rank=dp_rank, done=False,
                  status="starting", logical_to_global=config["logical_to_global"], steps=[])
    context = dict(step=None, phase="setup")
    dist = recorder = None
    control = None
    def save():
        _write(output / f"rank{rank}.json", record)
    try:
        save()
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=init_method, rank=rank,
                                world_size=config["world_size"], timeout=timedelta(seconds=15))
        control = dist.new_group(list(range(config["world_size"])), timeout=timedelta(seconds=30))
        cp_group = dp_group = None
        order = config["logical_to_global"]
        for d in range(config["dp"]):
            members = order[d * config["cp"]:(d + 1) * config["cp"]]
            group = dist.new_group(members, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
            if rank in members:
                cp_group = group
        for c in range(config["cp"]):
            members = order[c::config["cp"]]
            group = dist.new_group(members, timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
            if rank in members:
                dp_group = group
        if dist.get_rank(cp_group) != cp_rank or dist.get_rank(dp_group) != dp_rank:
            raise RuntimeError("native subgroup rank does not match requested logical coordinates")
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=config["cp"], group=cp_group),
                                intra_dp=SimpleNamespace(rank=dp_rank, size=config["dp"], group=dp_group),
                                intra_dp_cp=SimpleNamespace(group=dist.group.WORLD))
        namespace = dict(torch=torch, dist=dist, F=F, get_parallel_state=lambda: state,
                         Namespace=SimpleNamespace, RolloutBatch=dict, _diag_logger=logging.getLogger(__name__))
        if config["trace"]:
            recorder = _Trace(output / f"rank{rank}.trace.jsonl", rank, dist, context)
        _load_native(source, namespace, recorder)
        native_gae = namespace["get_advantages_and_returns_batch"]
        native_whiten = namespace["distributed_masked_whiten"]
        step_record = None

        def gae(*args, **kwargs):
            previous, context["phase"] = context["phase"], "gae"
            try:
                # Record the rewards actually supplied by the candidate caller.
                step_record["gae_inputs"] = dict(
                    values=[x.tolist() for x in (args[2] if len(args) > 2 else kwargs["values_list"])],
                    rewards=[x.tolist() for x in (args[3] if len(args) > 3 else kwargs["rewards_list"])])
                save()
                result = native_gae(*args, **kwargs)
                step_record["raw_advantages"] = [x.tolist() for x in result[0]]
                step_record["raw_returns"] = [x.tolist() for x in result[1]]
                save()
                return result
            finally:
                context["phase"] = previous

        def whiten(values, mask, *args, **kwargs):
            previous, context["phase"] = context["phase"], "whiten"
            capture = dict(values=values.tolist(), masks=mask.tolist())
            step_record["whitening"] = capture
            save()
            try:
                result = native_whiten(values, mask, *args, **kwargs)
                capture["output"] = result.tolist()
                save()
                return result
            except Exception as exc:
                capture.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                context["phase"] = previous

        namespace["get_advantages_and_returns_batch"] = gae
        namespace["distributed_masked_whiten"] = whiten
        settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator="ppo",
                                   gamma=1.0, lambd=1.0, gae_lambda_k=0.5,
                                   normalize_advantages=True, qkv_format="thd")
        record["status"] = "running"
        for update, length in enumerate(config["steps"]):
            context.update(step=update, phase="slice")
            samples = _samples(length, config["cp"], dp_rank, config["seed"], update, config["mask_mode"])
            step_record = dict(step=update, response_length=length, samples=samples, status="running")
            record["steps"].append(step_record)
            save()
            try:
                values = [namespace["slice_log_prob_with_cp"](
                    torch.tensor(s["values"], dtype=torch.float32), s["total"], s["length"]) for s in samples]
                step_record["sliced_values"] = [x.tolist() for x in values]
                batch = dict(values=values, log_probs=[torch.zeros_like(x) for x in values],
                             rewards=[s["reward"] for s in samples], total_lengths=[s["total"] for s in samples],
                             response_lengths=[s["length"] for s in samples],
                             loss_masks=[torch.tensor(s["masks"], dtype=torch.float32) for s in samples])
                context["phase"] = "compute"
                namespace["compute_advantages_and_returns"](settings, batch)
                step_record.update(status="completed", advantages=[x.tolist() for x in batch["advantages"]],
                                   returns=[x.tolist() for x in batch["returns"]])
            except Exception as exc:
                step_record.update(status="candidate_error", error_type=type(exc).__name__, error=str(exc))
            save()
            # Even a rank that returned or raised stays alive here while its
            # peers finish or reach the genuine data-group timeout.
            context["phase"] = "control"
            # Tensor-only synchronization avoids all_gather_object's NumPy
            # dependency in otherwise standalone CPU PyTorch environments.
            failed = torch.tensor([int(step_record["status"] != "completed")], dtype=torch.int32)
            dist.all_reduce(failed, group=control)
            if failed.item():
                record["status"] = "candidate_error" if step_record["status"] != "completed" else "peer_error"
                record["stopped_after"] = update
                break
        else:
            record["status"] = "completed"
        record["done"] = True
        save()
    except Exception as exc:
        record.update(status="infrastructure_error", error_type=type(exc).__name__, error=str(exc))
        save()
    finally:
        if recorder is not None:
            dist.all_reduce = recorder.native_all_reduce
            recorder.stream.close()
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
    return 2 if record["status"] == "infrastructure_error" else 1 if record["status"] != "completed" else 0


def run_probe(source, output, *, cp=2, dp=2, order=None, steps=(11, 3), mask_mode="ones",
              seed=DEFAULT_SEED, trace=False):
    config = _configuration(cp, dp, order, steps, mask_mode, seed, trace)
    output, source = Path(output).absolute(), Path(source).absolute()
    output.mkdir(parents=True, exist_ok=False)
    config.update(source=str(source), output=str(output))
    _write(output / "config.json", config)
    summary = dict(schema_version=1, status="infrastructure_error", exit_code=2,
                   config=config, outer_timeout=False, ranks=[])
    processes, logs = [], []
    started = time.monotonic()
    try:
        _, summary["source_sha256"] = _source_functions(source)
        import torch.distributed as dist
        if sys.version_info < (3, 10):
            raise RuntimeError("Python >= 3.10 is required to execute native functions")
        if not dist.is_available() or not dist.is_gloo_available():
            raise RuntimeError("CPU PyTorch with Gloo is required")
        init_method = (output / "rendezvous").as_uri()
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCH_DISTRIBUTED_DEBUG="OFF")
        for rank in range(config["world_size"]):
            log = (output / f"rank{rank}.log").open("w")
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(rank),
                       "--_config", str(output / "config.json"), "--_init-method", init_method]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment))
        deadline = started + OUTER_TIMEOUT
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                summary["outer_timeout"] = True
                summary["error"] = "worker launch exceeded the 45 second outer deadline"
                break
            time.sleep(0.05)
    except Exception as exc:
        summary.update(error_type=type(exc).__name__, error=str(exc))
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
    if "source_sha256" in summary:
        try:
            summary["source_sha256_after"] = {
                relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
                for relative in summary["source_sha256"]
            }
            summary["source_changed"] = sorted(
                relative for relative, digest in summary["source_sha256"].items()
                if summary["source_sha256_after"][relative] != digest)
            if summary["source_changed"]:
                summary.update(error_type="SourceChangedError", error="candidate source changed during execution")
        except OSError as exc:
            summary.update(error_type=type(exc).__name__, error=f"cannot recheck candidate source after execution: {exc}")
    for rank, process in enumerate(processes):
        path = output / f"rank{rank}.json"
        try:
            row = json.loads(path.read_text())
            if row.get("rank") != rank or not row.get("done"):
                raise ValueError("worker did not complete its status protocol")
            summary["ranks"].append(dict(rank=rank, status=row["status"], exit_code=process.returncode,
                                         steps_completed=sum(s["status"] == "completed" for s in row["steps"])))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["ranks"].append(dict(rank=rank, status="infrastructure_error",
                                         exit_code=process.returncode, error=str(exc)))
    if (not summary.get("error") and not summary["outer_timeout"]
            and len(summary["ranks"]) == config["world_size"]
            and all(r["status"] in ("completed", "candidate_error", "peer_error")
                    and r["exit_code"] == (0 if r["status"] == "completed" else 1) for r in summary["ranks"])):
        failed = any(r["status"] != "completed" for r in summary["ranks"])
        summary.update(status="candidate_error" if failed else "completed", exit_code=int(failed))
    summary["elapsed"] = time.monotonic() - started
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, help="fresh output directory")
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--order", help="logical-to-global permutation; sorted rows and columns required")
    parser.add_argument("--steps", default="11,3", help="up to 16 comma-separated response lengths, each 1..4096")
    parser.add_argument("--mask-mode", choices=("ones", "sparse"), default="ones")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--_worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_init-method", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        return _worker(json.loads(args._config.read_text()), args._worker, args._init_method)
    if args.output is None:
        parser.error("--output is required")
    try:
        summary = run_probe(args.source, args.output, cp=args.cp, dp=args.dp, order=args.order,
                            steps=args.steps, mask_mode=args.mask_mode, seed=args.seed, trace=args.trace)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps(dict(status="infrastructure_error", exit_code=2, error=str(exc))), file=sys.stderr)
        return 2
    print(json.dumps(summary, allow_nan=False))
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
