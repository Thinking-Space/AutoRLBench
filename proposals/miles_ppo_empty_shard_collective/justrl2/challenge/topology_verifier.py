#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only native PPO/CP/DP verifier (Python >= 3.10, CPU PyTorch/Gloo).

CLI: --source ROOT_CONTAINING_MILES --output FRESH_DIRECTORY
API: run_suite(source, output, *, topologies=None, scenarios=None,
               collective_timeout=2.0, outer_timeout=45.0) -> JSON-compatible dict.
Exit codes: 0 pass, 1 candidate failure, 2 infrastructure/source-loading error.
Each topology pools its scenarios in one subprocess launch. A poisoned collective
stops that topology; missing cases cannot pass. This is trusted-code execution,
not a sandbox for untrusted submissions. Do not ship this file to the solver.
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
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

FUNCTIONS = {
    "miles/backends/training_utils/cp_utils.py": (
        "_parallel_rank", "_parallel_size", "get_logits_and_tokens_offset_with_cp",
        "local_response_to_full", "slice_log_prob_with_cp"),
    "miles/utils/ppo_utils.py": (
        "length_adaptive_lambda", "get_advantages_and_returns_batch", "chunked_gae", "vanilla_gae"),
    "miles/utils/distributed_utils.py": ("distributed_masked_whiten",),
    "miles/backends/training_utils/loss.py": ("compute_advantages_and_returns",),
}
TOPOLOGIES = {
    "cp2_dp2": (2, [0, 1, 2, 3]),
    "cp4_dp2": (4, list(range(8))),
    "cp1_dp2": (1, [0, 1]),
    # Both axes retain native subgroup ranks, but global_rank % CP is wrong.
    "cp4_dp2_permuted": (4, [0, 2, 4, 6, 1, 3, 5, 7]),
}
SCENARIOS = ("baseline", "uneven_counts", "zero_masks_nonempty", "recorded_geometry",
             "single_valid_token", "asymmetric_empty", "empty_dp_replica", "empty_dp_group", "global_zero_mask")
COVERAGE_LIMITS = (
    "AST-selected candidate functions run unchanged: CP slicing/reconstruction, native PPO "
    "get_advantages_and_returns_batch, compute_advantages_and_returns and distributed_masked_whiten. "
    "Synthetic float32 critic values; thd, gamma=1, length-adaptive lambda k=0.5, KL=0. "
    "Real CPU/Gloo CP and DP groups. Independent scalar GAE and two-pass sample-variance oracle. "
    "AST scope excludes module initialization, models, Megatron, GPU/NCCL, forward/backward, "
    "optimizer/checkpoints and full training. GPU rerun and calibration remain pending."
)
ZERO_ERROR = "The global mask sum across all participating GPUs is zero."


def _write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _safe(value):
    """Preserve nonfinite evidence in strict JSON; assessment rejects these strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return [_safe(x) for x in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    return value


def _source_nodes(source):
    selected, hashes = {}, {}
    for relative, required in FUNCTIONS.items():
        path = source / relative
        text = path.read_text()
        hashes[relative] = hashlib.sha256(text.encode()).hexdigest()
        tree = ast.parse(text, filename=str(path))
        def bound(node):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                return {node.name}
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return {a.asname or a.name.split('.')[0] for a in node.names}
            if isinstance(node, ast.Assign):
                return {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            return set()
        by_name = {name: node for node in tree.body for name in bound(node)}
        if not set(required) <= set(by_name):
            raise ValueError(f'{relative}: missing native functions')
        chosen = {by_name[name] for name in required}
        provided = set(namespace_name for names in FUNCTIONS.values() for namespace_name in names)
        provided.update({'torch', 'dist', 'F', 'get_parallel_state', 'Namespace', 'RolloutBatch', '_diag_logger'})
        native_modules = {name[:-3].replace('/', '.') for name in FUNCTIONS}
        while True:
            needed = {n.id for node in chosen for n in ast.walk(node)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            extra = {by_name[name] for name in needed if name in by_name
                     and (name not in provided or isinstance(by_name[name], (ast.FunctionDef, ast.ClassDef)))
                     and not (isinstance(by_name[name], ast.ImportFrom)
                              and (by_name[name].module in native_modules
                                   or by_name[name].module in ('parallel', 'miles.backends.training_utils.parallel')))}
            if extra <= chosen:
                break
            chosen |= extra
        selected[relative] = [n for n in tree.body if n in chosen]
    helper = source / 'miles/utils/moments.py'
    if helper.exists():
        hashes['miles/utils/moments.py'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    return selected, hashes


def _load(source, namespace):
    selected, _ = _source_nodes(source)
    for relative, nodes in selected.items():
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source / relative), "exec"), namespace)
        if relative.endswith("cp_utils.py"):
            for name in ('miles', 'miles.utils', 'miles.backends', 'miles.backends.training_utils'):
                package = ModuleType(name)
                package.__path__ = [str(source.joinpath(*name.split('.')))]
                sys.modules[name] = package
            sys.modules.pop('miles.utils.moments', None)
            cp_module = ModuleType("miles.backends.training_utils.cp_utils")
            cp_module.__dict__.update(namespace)
            sys.modules[cp_module.__name__] = cp_module


def _indices(total, length, cp_size, cp_rank):
    """Independent token ownership oracle; never used to feed candidate shards."""
    width = math.ceil(total / (2 * cp_size))
    return [j for j in range(length)
            if cp_size == 1 or min((total - length - 1 + j) // width,
                                  2 * cp_size - 1 - (total - length - 1 + j) // width) == cp_rank]


def _samples(name, cp_size, dp_rank):
    samples = []
    for sample in range(1 if name == "recorded_geometry" else 2):
        total = cp_size * (8 + 2 * sample)
        length = total - 2 - sample - dp_rank
        if name == "recorded_geometry":
            # Original replay lengths exercise ceil-to-chunk padding. Values
            # remain synthetic: this is geometry coverage, not a model replay.
            total, length = ((424, 2), (357, 329))[dp_rank]
        if name == "uneven_counts":
            length -= 2 * dp_rank + sample
        if name == "asymmetric_empty" and dp_rank == 0 or name == "empty_dp_group":
            length = 1 + sample
        if name == "empty_dp_replica" and dp_rank == 0:
            length = 0
        values = [0.13 * dp_rank + 0.07 * sample + 0.03 * t + 0.011 * ((t * t + 3) % 7)
                  for t in range(length)]
        masks = [float(not (name == "uneven_counts" and (t + dp_rank) % 3 == 0))
                 for t in range(length)]
        if name == "global_zero_mask" or name == "zero_masks_nonempty" and dp_rank == 0:
            masks = [0.0] * length
        if name == "single_valid_token":
            masks = [0.0] * length
            if dp_rank == 0 and sample == 0:
                for cp_rank in range(cp_size):
                    owned = _indices(total, length, cp_size, cp_rank)
                    if not owned:
                        raise ValueError("Singleton fixture needs a nonempty shard at every CP coordinate")
                    masks[owned[0]] = 1.0
        reward = ((-0.7, 0.3), (1.5, -0.2))[dp_rank][sample]
        samples.append(dict(total=total, length=length, values=values, masks=masks, reward=reward))
    return samples


def _reference(name, cp_size, mapping):
    rows = {}
    for logical, rank in enumerate(mapping):
        dp_rank, cp_rank = divmod(logical, cp_size)
        row = dict(values=[], raw=[], returns=[], masks=[])
        for sample in _samples(name, cp_size, dp_rank):
            values, length = sample["values"], sample["length"]
            raw = [0.0] * length
            decay = 0.5 ** (1 / length) if length else 0.0
            carry = 0.0
            for t in reversed(range(length)):
                reward = sample["reward"] if t == length - 1 else 0.0
                next_value = values[t + 1] if t + 1 < length else 0.0
                carry = reward + next_value - values[t] + decay * carry
                raw[t] = carry
            indices = _indices(sample["total"], length, cp_size, cp_rank)
            row["values"].append([values[i] for i in indices])
            row["raw"].append([raw[i] for i in indices])
            row["returns"].append([sample["reward"] for _ in indices])
            row["masks"].extend(sample["masks"][i] for i in indices)
        row["group"] = sorted(mapping[d * cp_size + cp_rank] for d in range(2))
        rows[rank] = row
    stats = []
    for cp_rank in range(cp_size):
        members = sorted(mapping[d * cp_size + cp_rank] for d in range(2))
        selected = [a for r in members for a, m in zip(_flat(rows[r]["raw"]), rows[r]["masks"]) if m]
        count = len(selected)
        mean = math.fsum(selected) / count if count else None
        variance = math.fsum((a - mean) ** 2 for a in selected) / (count - 1) if count > 1 else 0.0
        stats.append(dict(members=members, count=count, mean=mean, sample_variance=variance))
        for rank in members:
            rows[rank]["count"] = count
            rows[rank]["normalized"] = (
                [[(a - mean) / math.sqrt(variance + 1e-8) for a in chunk] for chunk in rows[rank]["raw"]]
                if count else None)
    return rows, stats


def _flat(chunks):
    return [x for chunk in chunks for x in chunk]


def _worker(args):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    torch.set_num_threads(1)
    cp_size, mapping = TOPOLOGIES[args.topology]
    rank = args.rank
    logical = mapping.index(rank)
    dp_rank, cp_rank = divmod(logical, cp_size)
    output = args.output
    record = dict(rank=rank, cp_rank=cp_rank, dp_rank=dp_rank, done=False, cases=[])
    case = None
    stage = "compute"
    def save():
        _write(output / f"rank{rank}.json", _safe(record))
    def event(kind, **fields):
        case["events"].append(dict(event=kind, stage=stage, **fields))
        save()
    def members(group):
        return dist.get_process_group_ranks(group if group is not None else dist.group.WORLD)
    class RecordingDist:
        def __getattr__(self, name):
            return getattr(dist, name)
        def all_reduce(self, tensor, *positional, **kwargs):
            group = kwargs.get("group", positional[1] if len(positional) > 1 else None)
            group_members = members(group)
            call = len(case["events"])
            start = time.monotonic()
            event("collective_enter", call=call, members=group_members, shape=list(tensor.shape))
            try:
                result = dist.all_reduce(tensor, *positional, **kwargs)
            except Exception as exc:
                event("collective_error", call=call, members=group_members,
                      error=str(exc), error_type=type(exc).__name__, elapsed=time.monotonic() - start)
                raise
            event("collective_return", call=call, members=group_members)
            return result
    try:
        save()
        dist.init_process_group("gloo", init_method=args.init_method, rank=rank,
                                world_size=len(mapping), timeout=timedelta(seconds=15))
        control = dist.new_group(list(range(len(mapping))), timeout=timedelta(seconds=15))
        cp_group = dp_group = None
        for d in range(2):
            group_members = sorted(mapping[d * cp_size + c] for c in range(cp_size))
            group = dist.new_group(group_members, timeout=timedelta(seconds=args.collective_timeout))
            if rank in group_members:
                cp_group = group
        for c in range(cp_size):
            group_members = sorted(mapping[d * cp_size + c] for d in range(2))
            group = dist.new_group(group_members, timeout=timedelta(seconds=args.collective_timeout))
            if rank in group_members:
                dp_group = group
        assert dist.get_rank(cp_group) == cp_rank and dist.get_rank(dp_group) == dp_rank
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size, group=cp_group),
                                intra_dp=SimpleNamespace(rank=dp_rank, size=2, group=dp_group),
                                intra_dp_cp=SimpleNamespace(group=dist.group.WORLD))
        namespace = dict(torch=torch, dist=RecordingDist(), F=F, get_parallel_state=lambda: state,
                         Namespace=SimpleNamespace, RolloutBatch=dict, _diag_logger=logging.getLogger(__name__))
        _load(args.source, namespace)
        native_gae = namespace["get_advantages_and_returns_batch"]
        native_whiten = namespace["distributed_masked_whiten"]
        def gae(*positional, **kwargs):
            nonlocal stage
            previous, stage = stage, "gae"
            event("gae_enter")
            try:
                result = native_gae(*positional, **kwargs)
                event("gae_return", advantages=[x.tolist() for x in result[0]],
                      returns=[x.tolist() for x in result[1]])
                return result
            finally:
                stage = previous
        def whiten(values, mask, process_group=None, **kwargs):
            nonlocal stage
            previous, stage = stage, "whiten"
            event("whiten_enter", values=values.tolist(), masks=mask.tolist(), members=members(process_group))
            try:
                result = native_whiten(values, mask, process_group=process_group, **kwargs)
                event("whiten_return", values=result.tolist())
                return result
            except Exception as exc:
                event("whiten_error", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                stage = previous
        namespace["get_advantages_and_returns_batch"] = gae
        namespace["distributed_masked_whiten"] = whiten
        for scenario in args.scenarios.split(","):
            case = dict(scenario=scenario, events=[], status="running")
            record["cases"].append(case)
            save()
            try:
                samples = _samples(scenario, cp_size, dp_rank)
                values = [namespace["slice_log_prob_with_cp"](
                    torch.tensor(s["values"], dtype=torch.float32), s["total"], s["length"]) for s in samples]
                case["sliced_values"] = [x.tolist() for x in values]
                batch = dict(values=values, log_probs=[torch.zeros_like(x) for x in values],
                             rewards=[s["reward"] for s in samples], total_lengths=[s["total"] for s in samples],
                             response_lengths=[s["length"] for s in samples],
                             loss_masks=[torch.tensor(s["masks"], dtype=torch.float32) for s in samples])
                settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator="ppo",
                                           gamma=1.0, lambd=1.0, gae_lambda_k=0.5,
                                           normalize_advantages=True, qkv_format="thd")
                namespace["compute_advantages_and_returns"](settings, batch)
                case.update(status="ok", advantages=[x.tolist() for x in batch["advantages"]],
                            returns=[x.tolist() for x in batch["returns"]])
            except Exception as exc:
                case.update(status="error", error_type=type(exc).__name__, error=str(exc))
            save()
            poisoned = any(e["event"] == "collective_error" for e in case["events"])
            flags = [None] * len(mapping)
            # A separate group keeps successful/skipping peers alive while a native
            # collective times out; a peer closing its socket is not our witness.
            dist.all_gather_object(flags, poisoned, group=control)
            if any(flags):
                record["aborted_after"] = scenario
                break
        record["done"] = True
        save()
    except Exception as exc:
        record["infrastructure_error"] = f"{type(exc).__name__}: {exc}"
        save()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _assess(topology, scenarios, rows, collective_timeout):
    cp_size, mapping = TOPOLOGIES[topology]
    failures, cases = [], []
    def fail(kind, scenario, rank, message, **fields):
        failures.append(dict(kind=kind, scenario=scenario, rank=rank, message=message, **fields))
    def compare(actual, expected, scenario, rank, field):
        if actual is None:
            fail("candidate_error", scenario, rank, f"Missing {field}")
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                fail("numerical_mismatch", scenario, rank, f"{field}: incorrect tensor/chunk shape")
                return
            for i, (a, b) in enumerate(zip(actual, expected)):
                compare(a, b, scenario, rank, f"{field}[{i}]")
            return
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            fail("nonfinite", scenario, rank, f"{field}: {actual!r}")
        elif not math.isclose(actual, expected, abs_tol=3e-4, rel_tol=3e-4):
            fail("numerical_mismatch", scenario, rank, f"{field}: got {actual}, expected {expected}")
    for scenario in scenarios:
        refs, stats = _reference(scenario, cp_size, mapping)
        before = len(failures)
        selected = {r["rank"]: next((c for c in r["cases"] if c["scenario"] == scenario), None) for r in rows}
        witnesses = []
        for rank, case in selected.items():
            if case is None:
                fail("incomplete_suite", scenario, rank, "Case not executed after collective failure")
                continue
            ref = refs[rank]
            events = case["events"]
            gae = [e for e in events if e["event"] == "gae_return"]
            whitening = [e for e in events if e["event"] == "whiten_enter"]
            compare(case.get("sliced_values"), ref["values"], scenario, rank, "native CP slices")
            if len(gae) != 1:
                fail("candidate_error", scenario, rank, "Native PPO GAE did not return exactly once")
            else:
                compare(gae[0]["advantages"], ref["raw"], scenario, rank, "raw PPO advantages")
                compare(gae[0]["returns"], ref["returns"], scenario, rank, "PPO returns")
            if len(whitening) != 1:
                fail("candidate_error", scenario, rank, "Native distributed whitening was skipped or repeated")
            else:
                compare(whitening[0]["values"], _flat(ref["raw"]), scenario, rank, "whitening input")
                compare(whitening[0]["masks"], ref["masks"], scenario, rank, "CP loss masks")
                if whitening[0]["members"] != ref["group"]:
                    fail("wrong_process_group", scenario, rank, "Whitening must use the intra-DP group")
            for e in events:
                if e["event"] == "collective_enter" and e["stage"] == "whiten" and e["members"] != ref["group"]:
                    fail("wrong_process_group", scenario, rank, "Whitening collective has wrong members")
            if ref["count"] == 0:
                errors = [e for e in events if e["event"] == "whiten_error"]
                if not (case["status"] == "error" and case.get("error_type") == "ValueError"
                        and case.get("error") == ZERO_ERROR and len(errors) == 1
                        and errors[0]["error_type"] == "ValueError"
                        and any(e["event"] == "collective_return" and e["stage"] == "whiten" for e in events)):
                    fail("expected_zero_count_error", scenario, rank,
                         "Every zero-count DP group member must observe the native ValueError after its collective")
            elif case["status"] != "ok":
                fail("candidate_error", scenario, rank, case.get("error", "Incomplete candidate invocation"))
            else:
                compare(case.get("advantages"), ref["normalized"], scenario, rank, "normalized advantages")
                compare(case.get("returns"), ref["returns"], scenario, rank, "preserved returns")
            for e in events:
                if not (e["event"] == "collective_error" and e["stage"] == "whiten"
                        and e["members"] == ref["group"] and "gloo" in e["error"].lower()
                        and "timed out" in e["error"].lower() and e["elapsed"] >= collective_timeout * 0.8):
                    continue
                entry = next((x for x in events if x["event"] == "collective_enter" and x["call"] == e["call"]), None)
                if not entry or entry["shape"] != [3] or ref["count"] == 0:
                    continue
                for peer in ref["group"]:
                    skipped = selected[peer]
                    if (peer != rank and skipped and skipped["status"] == "ok"
                            and sum(refs[peer]["masks"]) == 0
                            and any(x["event"] == "gae_return" for x in skipped["events"])
                            and not any(x["event"] == "whiten_enter" for x in skipped["events"])):
                        witnesses.append(dict(waiting_rank=rank, skipped_rank=peer, dp_members=ref["group"],
                                              elapsed=e["elapsed"], native_error=e["error"]))
        if witnesses:
            fail("collective_participation_mismatch", scenario, None,
                 "Native DP whitening timed out while a zero-contribution peer returned without calling it",
                 witnesses=witnesses)
        cases.append(dict(scenario=scenario, passed=len(failures) == before,
                          expected_zero_count_ranks=[r for r in mapping if refs[r]["count"] == 0],
                          reference_group_stats=stats))
    return cases, failures


def _launch(source, output, topology, scenarios, collective_timeout, outer_timeout):
    cp_size, mapping = TOPOLOGIES[topology]
    report = dict(topology=topology, cp_size=cp_size, dp_size=2, world_size=len(mapping),
                  logical_to_global=mapping, cases=[], failures=[])
    output.mkdir()
    processes, streams = [], []
    started = time.monotonic()
    infrastructure = []
    try:
        for rank in range(len(mapping)):
            stream = (output / f"rank{rank}.log").open("w")
            streams.append(stream)
            command = [sys.executable, str(Path(__file__).resolve()), "--source", str(source),
                       "--output", str(output), "--rank", str(rank), "--topology", topology,
                       "--scenarios", ",".join(scenarios), "--collective-timeout", str(collective_timeout),
                       "--init-method", (output / "rendezvous").as_uri()]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                              env=env, start_new_session=True))
        for process in processes:
            process.wait(timeout=max(0.001, outer_timeout - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        infrastructure.append("Outer timeout; incomplete runs never establish reproduction")
    except Exception as exc:
        infrastructure.append(f"Launch error: {exc}")
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        for stream in streams:
            stream.close()
    rows = []
    for rank in range(len(mapping)):
        try:
            row = json.loads((output / f"rank{rank}.json").read_text())
            if row["rank"] != rank or not row["done"] or row.get("infrastructure_error"):
                raise ValueError(row.get("infrastructure_error", "Missing completed rank record"))
            if rank >= len(processes) or processes[rank].returncode != 0:
                raise ValueError("Worker process exited unsuccessfully")
            rows.append(row)
        except Exception as exc:
            infrastructure.append(f"Rank {rank}: {exc}")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if infrastructure:
        report["failures"] = [dict(kind="infrastructure_error", message=message) for message in infrastructure]
        report["status"] = "infrastructure_error"
    else:
        try:
            report["cases"], report["failures"] = _assess(topology, scenarios, rows, collective_timeout)
            report["status"] = "fail" if report["failures"] else "pass"
        except Exception as exc:
            report["status"] = "infrastructure_error"
            report["failures"] = [dict(kind="infrastructure_error", message=f"Invalid evidence: {exc}")]
    return report


def run_suite(source, output, *, topologies=None, scenarios=None, collective_timeout=2.0, outer_timeout=45.0):
    """Execute native source. Results always fail closed; output must not exist.

    Default covers all topology/scenario pairs. Subsets are for mutation/unit
    tests; ``full_suite`` explicitly identifies reduced runs. Per-topology outer
    deadline covers startup and all scenarios, including subprocess cleanup.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    topologies = list(TOPOLOGIES) if topologies is None else list(topologies)
    scenarios = list(SCENARIOS) if scenarios is None else list(scenarios)
    result = dict(status="infrastructure_error", passed=False, observed_intended_mismatch=False,
                  coverage_limits=COVERAGE_LIMITS, gpu_training_validated=False,
                  source=str(source), source_sha256={}, topologies=[],
                  full_suite=topologies == list(TOPOLOGIES) and scenarios == list(SCENARIOS))
    try:
        if not topologies or not scenarios or len(set(topologies)) != len(topologies) or len(set(scenarios)) != len(scenarios):
            raise ValueError("Select nonempty, unique topology and scenario names")
        if set(topologies) - TOPOLOGIES.keys() or set(scenarios) - set(SCENARIOS):
            raise ValueError("Unknown topology or scenario")
        if not (math.isfinite(collective_timeout) and collective_timeout > 0
                and math.isfinite(outer_timeout) and outer_timeout > 0):
            raise ValueError("Timeouts must be positive finite seconds")
        if sys.version_info < (3, 10):
            raise RuntimeError("Native source requires Python >= 3.10 (3.12 recommended)")
        _, hashes = _source_nodes(source)
        result["source_sha256"] = hashes
        for topology in topologies:
            report = _launch(source, output / topology, topology, scenarios, collective_timeout, outer_timeout)
            result["topologies"].append(report)
            if report["status"] == "infrastructure_error":
                break
        if _source_nodes(source)[1] != hashes:
            raise RuntimeError("Candidate source changed during verification")
        statuses = [r["status"] for r in result["topologies"]]
        result["status"] = ("infrastructure_error" if "infrastructure_error" in statuses else
                            "fail" if "fail" in statuses else "pass")
        result["passed"] = result["status"] == "pass"
        result["observed_intended_mismatch"] = result["status"] == "fail" and any(
            f["kind"] == "collective_participation_mismatch" for r in result["topologies"] for f in r["failures"])
    except Exception as exc:
        result.update(status="infrastructure_error", passed=False, observed_intended_mismatch=False,
                      error=f"{type(exc).__name__}: {exc}")
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collective-timeout", type=float, default=2.0)
    parser.add_argument("--outer-timeout", type=float, default=45.0)
    parser.add_argument("--topology", choices=list(TOPOLOGIES))
    parser.add_argument("--scenarios")
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--init-method", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rank is not None:
        _worker(args)
        return 0
    try:
        result = run_suite(args.source, args.output, topologies=[args.topology] if args.topology else None,
                           scenarios=args.scenarios.split(",") if args.scenarios else None,
                           collective_timeout=args.collective_timeout, outer_timeout=args.outer_timeout)
    except FileExistsError:
        print("Infrastructure error: --output must be a fresh directory", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("status", "passed", "observed_intended_mismatch", "full_suite")}))
    print(COVERAGE_LIMITS)
    return {"pass": 0, "fail": 1, "infrastructure_error": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
