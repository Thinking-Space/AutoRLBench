#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU check of exact public GAE/CP/whitening functions on the constructed replay.

Functions are compiled from the prepared source AST to avoid Megatron/FA imports.
This runs real Gloo groups and GAE, but it supplies synthetic critic values and
does not perform model forward/backward or validate full training.
"""
import argparse
import ast
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Missing pinned source function")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def worker(args):
    rank = args.rank
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=args.init_method, rank=rank, world_size=4,
                            timeout=timedelta(seconds=5))
    cp = dp = None
    for members in [[0, 1], [2, 3], [0, 2], [1, 3]]:
        group = dist.new_group(members, timeout=timedelta(seconds=5))
        if rank in members:
            if members[1] - members[0] == 1:
                cp = group
            else:
                dp = group
    state = SimpleNamespace(cp=SimpleNamespace(rank=rank % 2, size=2, group=cp),
                            intra_dp=SimpleNamespace(rank=rank // 2, size=2, group=dp))
    record = {"rank": rank, "cp_rank": rank % 2, "dp_rank": rank // 2, "events": []}
    def emit(unused, event, **fields):
        record["events"].append({"event": event, **fields})
        (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    ns = {"torch": torch, "dist": dist, "F": F, "get_parallel_state": lambda: state,
          "Namespace": SimpleNamespace, "RolloutBatch": dict,
          "_diag_logger": logging.getLogger("check"), "emit": emit}
    cp_path = args.source / "miles/backends/training_utils/cp_utils.py"
    functions(cp_path, ["_parallel_rank", "_parallel_size", "get_logits_and_tokens_offset_with_cp",
                        "local_response_to_full", "slice_log_prob_with_cp"], ns)
    for name in ["miles", "miles.backends", "miles.backends.training_utils"]:
        sys.modules[name] = ModuleType(name)
    cp_module = ModuleType("miles.backends.training_utils.cp_utils")
    cp_module.__dict__.update(ns)
    sys.modules[cp_module.__name__] = cp_module
    functions(args.source / "miles/utils/ppo_utils.py",
              ["length_adaptive_lambda", "get_advantages_and_returns_batch", "chunked_gae", "vanilla_gae"], ns)
    functions(args.source / "miles/utils/distributed_utils.py", ["distributed_masked_whiten"], ns)
    functions(args.source / "miles/backends/training_utils/loss.py", ["compute_advantages_and_returns"], ns)
    samples = json.loads(args.replay.read_text())["samples"]
    sample = samples[rank // 2]
    total, length = len(sample["tokens"]), sample["response_length"]
    full_values = torch.full((length,), 0.52)
    values = ns["slice_log_prob_with_cp"](full_values, total, length)
    batch = {"values": [values], "log_probs": [torch.zeros_like(values)],
             "rewards": [sample["reward"]], "total_lengths": [total],
             "response_lengths": [length], "loss_masks": [torch.ones(length)]}
    settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator="ppo",
                               gamma=1.0, lambd=1.0, gae_lambda_k=0.5,
                               normalize_advantages=True, qkv_format="thd")
    started = time.monotonic()
    try:
        ns["compute_advantages_and_returns"](settings, batch)
        # Returns are independently checkable terminal-reward suffix sums.
        torch.testing.assert_close(batch["returns"][0], torch.full_like(values, sample["reward"]))
        record.update(status="ok", elapsed=time.monotonic() - started,
                      advantages=batch["advantages"][0].tolist(), returns=batch["returns"][0].tolist())
    except Exception as exc:
        record.update(status="error", error=repr(exc), elapsed=time.monotonic() - started)
    (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    # Let the deliberately missing collective expire without a peer-close shortcut.
    if args.variant == "faulty":
        time.sleep(7)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--variant", choices=["healthy", "faulty"])
    parser.add_argument("--rank", type=int)
    parser.add_argument("--init-method")
    args = parser.parse_args()
    if args.rank is not None:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for variant in ["healthy", "faulty"]:
        out = (args.output / variant).resolve(); out.mkdir()
        procs, streams = [], []
        try:
            for rank in range(4):
                log = (out / f"rank{rank}.log").open("w"); streams.append(log)
                procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "--source", str((args.pair / variant).resolve()), "--variant", variant,
                    "--replay", str(args.replay.resolve()), "--output", str(out),
                    "--rank", str(rank), "--init-method", f"file://{out / 'rendezvous'}"],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            expires = time.monotonic() + 45
            for p in procs:
                p.wait(timeout=max(1, expires - time.monotonic()))
        finally:
            for p in procs:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            for log in streams:
                log.close()
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("Worker failure; inspect " + str(out))
        rows = [json.loads((out / f"rank{rank}.json").read_text()) for rank in range(4)]
        if variant == "healthy":
            if not all(r["status"] == "ok" for r in rows):
                raise RuntimeError("Intact public GAE/whitening failed")
            for cp_rank in range(2):
                selected = [rows[cp_rank], rows[cp_rank + 2]]
                values = [x for r in selected for x in r["advantages"]]
                tensor = torch.tensor(values)
                torch.testing.assert_close(tensor.mean(), torch.tensor(0.0), atol=2e-4, rtol=0)
                torch.testing.assert_close(tensor.var(), torch.tensor(1.0), atol=2e-4, rtol=0)
        else:
            if not (rows[1]["status"] == "ok" and rows[1]["advantages"] == []
                    and rows[3]["status"] == "error" and "timed out" in rows[3]["error"].lower()
                    and rows[3]["elapsed"] >= 4):
                raise RuntimeError("Did not reproduce the injected whitening mismatch")
        summary[variant] = rows
    (args.output / "RESULT.json").write_text(json.dumps({"scope": "public GAE and whitening only; synthetic critic values",
                                                        "results": summary}, indent=2) + "\n")
    print("PASS: actual public CP reconstruction + length-adaptive GAE + whitening; injected path times out")


if __name__ == "__main__":
    main()
