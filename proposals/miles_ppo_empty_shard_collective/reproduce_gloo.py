#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Differential CPU reproducer for one extracted historical collective defect.

This is reviewer evidence, not a full RL run or an AutoRLBench task runner.
Two real Gloo ranks model one DP group at fixed CP coordinate 1 of CP2.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import torch
import torch.distributed as dist

from historical_kernels import (
    normalize_original,
    normalize_repaired,
)

CASES = ("mixed", "nonempty", "zero_mask_nonempty", "all_empty")


def inputs(case, rank):
    # total_length=16, CP2 rank1 owns logits [4, 12). A 2-token
    # response occupies [13, 15), so this rank has no response slice.
    response_length = 2 if case == "all_empty" or (case == "mixed" and rank == 0) else 8
    full_mask = torch.ones(response_length)
    if case == "zero_mask_nonempty" and rank == 0:
        full_mask.zero_()
    # An 8-token response contributes 5 local response tokens on CP rank1.
    values = torch.empty(0) if response_length == 2 else torch.arange(5, dtype=torch.float32) + rank * 5
    return [values], [full_mask], [16], [response_length]


def worker(args):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=args.init_method, rank=args.rank, world_size=2,
        timeout=timedelta(seconds=args.collective_timeout),
    )
    normalize = normalize_original if args.variant == "original" else normalize_repaired
    advantages, masks, totals, responses = inputs(args.case, args.rank)
    record = {"rank": args.rank, "variant": args.variant, "case": args.case,
              "cp_rank": 1, "cp_size": 2, "local_numel": advantages[0].numel()}
    start = time.monotonic()
    try:
        result = normalize(advantages, masks, totals, responses)
        record.update(status="ok", values=result[0].tolist())
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc))
    record["elapsed_seconds"] = round(time.monotonic() - start, 4)
    Path(args.result).write_text(json.dumps(record, indent=2) + "\n")
    if args.variant == "original" and args.case == "mixed" and args.rank == 0:
        # Keep the skipped peer alive until the other rank's actual allreduce
        # times out. This avoids substituting a connection-closed error for a hang.
        time.sleep(args.collective_timeout + 1)
    dist.destroy_process_group()


def expected_values(case, rank):
    values = inputs(case, rank)[0][0]
    population = torch.cat([inputs(case, r)[0][0] for r in range(2)])
    if case == "zero_mask_nonempty":
        population = inputs(case, 1)[0][0]
    return (values - population.mean()) * torch.rsqrt(population.var(unbiased=True) + 1e-8)


def run_case(variant, case, timeout):
    with tempfile.TemporaryDirectory(prefix="cp-whiten-") as tmp:
        root = Path(tmp)
        procs = []
        streams = []
        for rank in range(2):
            log = (root / f"rank{rank}.log").open("w")
            streams.append(log)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--variant", variant, "--case", case, "--rank", str(rank),
                   "--init-method", f"file://{root / 'rendezvous'}",
                   "--result", str(root / f"rank{rank}.json"),
                   "--collective-timeout", str(timeout)]
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "OMP_NUM_THREADS": "1"}))
        try:
            for proc in procs:
                proc.wait(timeout=timeout + 25)
            for log in streams:
                log.flush()
            for rank, proc in enumerate(procs):
                if proc.returncode != 0:
                    raise AssertionError(f"Worker {rank} failed:\n{(root / f'rank{rank}.log').read_text()}")
            records = [json.loads((root / f"rank{r}.json").read_text()) for r in range(2)]
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for log in streams:
                log.close()
    if variant == "original" and case == "mixed":
        assert records[0]["status"] == "ok" and records[0]["local_numel"] == 0, records
        assert records[1]["status"] == "error", records
        assert "timed out" in records[1]["error"].lower(), records
        assert records[1]["elapsed_seconds"] >= timeout * 0.8, records
        outcome = "collective_timeout_reproduced"
    elif variant == "repaired" and case == "all_empty":
        assert all(r["status"] == "error" and r["error_type"] == "ValueError"
                   and "global mask sum" in r["error"] for r in records), records
        outcome = "global_empty_mask_rejected"
    else:
        assert all(r["status"] == "ok" for r in records), records
        if case == "all_empty":
            assert all(r["values"] == [] for r in records), records
            outcome = "original_silently_skips_global_empty_validation"
        else:
            for rank, record in enumerate(records):
                torch.testing.assert_close(torch.tensor(record["values"]), expected_values(case, rank), rtol=1e-5, atol=1e-6)
            outcome = "completed_and_numerically_correct"
    return {"variant": variant, "case": case, "outcome": outcome, "ranks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=["original", "repaired"])
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--init-method")
    parser.add_argument("--result")
    parser.add_argument("--collective-timeout", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    cases = [args.case] if args.case else CASES
    variants = [args.variant] if args.variant else ["original", "repaired"]
    results = [run_case(variant, case, args.collective_timeout)
               for variant in variants for case in cases]
    output = {"python": platform.python_version(), "torch": torch.__version__,
              "backend": "gloo", "device": "cpu", "world_size": 2,
              "collective_timeout_seconds": args.collective_timeout,
              "results": results}
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
