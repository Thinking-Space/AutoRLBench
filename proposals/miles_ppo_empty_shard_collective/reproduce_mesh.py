#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run historical whitening in real CP x DP groups using NCCL or CPU Gloo.

The inputs are synthetic and this is not an end-to-end RL training replay.
The parent requires exact differential outcomes, including a target-stage
collective timeout for original/mixed. An outer deadline alone never passes.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import historical_kernels as kernels


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def full_input(case, dp_rank, device):
    length = 2 if case == "mixed" and dp_rank == 0 else 24
    # Keep synthetic advantages near zero. A large constant offset stresses the
    # historical E[x²]-E[x]² float32 variance separately from this liveness defect.
    values = torch.arange(length, dtype=torch.float32, device=device).remainder(7) - 3 + dp_rank
    mask = torch.ones(length, device=device)
    if case == "zero_mask_nonempty" and dp_rank == 0:
        mask.zero_()
    return values, mask


def response_indices(total, length, cp_rank, cp_size):
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
    _, _, _, offsets = kernels.get_logits_and_tokens_offset_with_cp(total, length, state)
    prompt = total - length
    return [i for start, end in offsets for i in range(max(0, start - prompt), max(0, end - prompt))]


def reference(case, cp_rank, cp_size):
    values, masks = [], []
    for dp_rank in range(2):
        v, m = full_input(case, dp_rank, "cpu")
        indices = response_indices(32, len(v), cp_rank, cp_size)
        values.append(v[indices])
        masks.append(m[indices])
    selected = torch.cat(values)[torch.cat(masks).bool()]
    if selected.numel() == 0:
        raise RuntimeError("Reference group is globally empty")
    variance = selected.var(unbiased=selected.numel() >= 2)
    return [(v - selected.mean()) * torch.rsqrt(variance + 1e-8) for v in values]


def worker(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != args.cp_size * 2:
        raise RuntimeError("Expected CP x DP2 world size")
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cp_rank, dp_rank = rank % args.cp_size, rank // args.cp_size
    record = {"rank": rank, "local_rank": local_rank, "cp_rank": cp_rank, "dp_rank": dp_rank,
              "backend": args.backend, "variant": args.variant, "case": args.case,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "nccl": list(torch.cuda.nccl.version()) if args.backend == "nccl" else None,
              "device": torch.cuda.get_device_name(device) if args.backend == "nccl" else "cpu"}
    path = args.output / f"rank{rank}.json"
    def stage(name, **fields):
        record.update(stage=name, **fields)
        save(path, record)
        print(json.dumps({"rank": rank, "stage": name}, sort_keys=True), flush=True)
    stage("initializing")
    dist.init_process_group(args.backend, init_method=args.init_method, rank=rank,
                            world_size=world, timeout=timedelta(seconds=max(60, args.collective_timeout)))
    cp_group = dp_group = None
    groups = []
    # Every world rank creates every group in identical order.
    for dp in range(2):
        members = list(range(dp * args.cp_size, (dp + 1) * args.cp_size))
        group = dist.new_group(members, backend=args.backend, timeout=timedelta(seconds=args.collective_timeout))
        if rank in members:
            cp_group = group
            record["cp_members"] = members
            groups.append(group)
    for cp in range(args.cp_size):
        members = [cp, cp + args.cp_size]
        group = dist.new_group(members, backend=args.backend, timeout=timedelta(seconds=args.collective_timeout))
        if rank in members:
            dp_group = group
            record["dp_members"] = members
            groups.append(group)
    stage("groups_created")
    # Validate both group families before attempting the defect.
    for group, members in [(cp_group, record["cp_members"]), (dp_group, record["dp_members"])]:
        value = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(value, group=group)
        torch.testing.assert_close(value.cpu(), torch.tensor([sum(r + 1.0 for r in members)]))
    values, mask = full_input(args.case, dp_rank, device)
    indices = response_indices(32, len(values), cp_rank, args.cp_size)
    local = values[indices].clone()
    reconstructed = torch.zeros_like(values)
    reconstructed[indices] = local
    dist.all_reduce(reconstructed, group=cp_group)
    torch.testing.assert_close(reconstructed, values)
    expected = reference(args.case, cp_rank, args.cp_size)[dp_rank]
    stage("before_barrier", group_smoke_passed=True, cp_reconstruction_passed=True,
          total_length=32, response_length=len(values), local_numel=local.numel(),
          response_indices=indices, local_values=local.cpu().tolist(),
          local_mask=mask[indices].cpu().tolist(), expected_values=expected.tolist())
    # World barrier only before the divergent path; no follow-up collective is
    # allowed to masquerade as a whitening timeout.
    dist.barrier()
    stage("before_whitening")
    fn = kernels.normalize_original if args.variant == "original" else kernels.normalize_repaired
    started = time.monotonic()
    try:
        result = fn([local], [mask], [32], [len(values)], cp_rank=cp_rank,
                    cp_size=args.cp_size, process_group=dp_group)[0]
        if args.backend == "nccl":
            torch.cuda.synchronize()
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)
        stage("whitening_complete", values=result.cpu().tolist(), numerical_match=True,
              elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        stage("whitening_error", error_type=type(exc).__name__, error=str(exc),
              elapsed_seconds=time.monotonic() - started)
        raise
    if args.variant == "original" and args.case == "mixed":
        # Keep ranks that skipped or completed their local group alive while
        # divergent peers time out, including the CP0 group which can complete.
        time.sleep(2 * args.collective_timeout + 5)
    for group in reversed(groups):
        dist.destroy_process_group(group)
    dist.destroy_process_group()


def run_case(args, variant, case):
    out = args.output / f"{variant}-{case}"
    out.mkdir(parents=True, exist_ok=False)
    world = args.cp_size * 2
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--backend", args.backend, "--cp-size", str(args.cp_size),
           "--variant", variant, "--case", case, "--output", str(out),
           "--collective-timeout", str(args.collective_timeout),
           "--init-method", f"file://{out / 'rendezvous'}"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "NCCL_DEBUG": "WARN",
           "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
           "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(out / "nccl_trace_rank_")}
    deadline = False
    processes, streams = [], []
    try:
        # Direct children, each with its own tracked process group. Avoid a
        # launcher whose independently sessioned children can escape cleanup.
        for rank in range(world):
            stream = (out / f"worker{rank}.log").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen(
                cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world)},
            ))
        expires = time.monotonic() + args.collective_timeout * 3 + 90
        while any(p.poll() is None for p in processes):
            if time.monotonic() >= expires:
                deadline = True
                break
            time.sleep(0.2)
    finally:
        for proc in processes:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        cleanup_deadline = time.monotonic() + 5
        while any(p.poll() is None for p in processes) and time.monotonic() < cleanup_deadline:
            time.sleep(0.1)
        for proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()
        for stream in streams:
            stream.close()
    returncodes = [p.returncode for p in processes]
    records = [json.loads(p.read_text()) for p in sorted(out.glob("rank*.json"))]
    rank_logs = {rank: (out / f"worker{rank}.log").read_text(errors="replace") for rank in range(world)}
    (out / "stdout.log").write_text("\n".join(f"--- worker {rank} ---\n{log}" for rank, log in rank_logs.items()))
    if deadline:
        raise RuntimeError(f"Outer deadline reached; not counted as reproduction: {out}")
    if len(records) != world or not all(r.get("group_smoke_passed") and r.get("cp_reconstruction_passed") for r in records):
        raise RuntimeError(f"Startup/topology validation incomplete: {out}")
    if variant == "original" and case == "mixed":
        waiting = [r for r in records if r["dp_rank"] == 1 and r["cp_rank"] > 0]
        skipped = [r for r in records if r["dp_rank"] == 0 and r["cp_rank"] > 0]
        completed_group = [r for r in records if r["cp_rank"] == 0]
        if not all(r["stage"] == "whitening_complete" and r["numerical_match"] for r in completed_group):
            raise RuntimeError(f"CP0 DP group did not complete correctly: {out}")
        if not all(r["local_numel"] == 0 and r["stage"] == "whitening_complete" for r in skipped):
            raise RuntimeError(f"Missing empty-shard skip evidence: {out}")
        if not all(r["local_numel"] > 0 and r["stage"] in ("before_whitening", "whitening_error") for r in waiting):
            raise RuntimeError(f"Missing blocked-peer evidence: {out}")
        if any(r["stage"] == "whitening_error" and not re.search(r"timed out|timeout", r.get("error", ""), re.I) for r in waiting):
            raise RuntimeError(f"Unrelated target-stage error: {out}")
        if args.backend == "nccl":
            pattern = re.compile(r"Watchdog caught collective operation timeout: WorkNCCL\([^\n]*OpType=ALLREDUCE, NumelIn=3, NumelOut=3,[^\n]*ran for (\d+) milliseconds before timing out")
            signature = any(
                any(int(m.group(1)) >= args.collective_timeout * 800 for m in pattern.finditer(rank_logs[r["rank"]]))
                for r in waiting
            )
        else:
            signature = any(r["stage"] == "whitening_error" and "timed out" in r.get("error", "").lower()
                            and r.get("elapsed_seconds", 0) >= args.collective_timeout * 0.8 for r in waiting)
        if all(code == 0 for code in returncodes) or not signature:
            raise RuntimeError(f"Missing target collective timeout: {out}")
        outcome = "target_collective_timeout_reproduced"
    else:
        if any(code != 0 for code in returncodes) or not all(r["stage"] == "whitening_complete" and r["numerical_match"] for r in records):
            raise RuntimeError(f"Expected clean numerical completion: {out}")
        outcome = "completed_and_numerically_correct"
    summary = {"variant": variant, "case": case, "outcome": outcome,
               "returncodes": returncodes, "ranks": records,
               "stdout_sha256": hashlib.sha256((out / "stdout.log").read_bytes()).hexdigest()}
    save(out / "RESULT.json", summary)
    print(json.dumps({k: summary[k] for k in ("variant", "case", "outcome", "returncodes")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--cp-size", type=int, choices=[2, 4], default=4)
    parser.add_argument("--collective-timeout", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--init-method")
    parser.add_argument("--variant", choices=["original", "repaired"])
    parser.add_argument("--case", choices=["mixed", "nonempty", "zero_mask_nonempty"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "nccl" and torch.cuda.device_count() < args.cp_size * 2:
        raise RuntimeError("Insufficient visible GPUs for requested mesh")
    cases = [("original", "nonempty"), ("repaired", "nonempty"),
             ("repaired", "zero_mask_nonempty"), ("repaired", "mixed"), ("original", "mixed")]
    if args.variant and args.case:
        cases = [(args.variant, args.case)]
    results = [run_case(args, variant, case) for variant, case in cases]
    save(args.output / "COMPLETE.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
        "cp_size": args.cp_size, "dp_size": 2, "world_size": args.cp_size * 2,
        "source_sha256": hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [{k: r[k] for k in ("variant", "case", "outcome", "returncodes")} for r in results],
    })
    print("MESH_REPRODUCTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
