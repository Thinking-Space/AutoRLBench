#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict evidence checks for the constructed replay case; no accuracy score."""
import argparse
import json
import math
import re
from pathlib import Path

from checkpoint_evidence import inspect_checkpoint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(root):
    events = []
    for path in sorted((root / "evidence").glob("*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    require(bool(events), "No per-rank evidence")
    return events


def grouped(events, event):
    result = {}
    for row in events:
        if row["event"] == event:
            key = row["role"], row["rank"]
            require(key not in result, f"Duplicate {event} record for {key}")
            result[key] = row
    return result


def verify_geometry(entries):
    expected = {(role, rank) for role in ["actor", "critic"] for rank in range(4)}
    require(set(entries) == expected, "Need all actor4 and critic4 pre-whitening records")
    for role in ["actor", "critic"]:
        groups = {}
        for (kind, rank), row in entries.items():
            if kind != role:
                continue
            require(row["cp_size"] == row["dp_size"] == 2, "Wrong CP/DP topology")
            require(rank in row["dp_members"] and rank in row["cp_members"], "Invalid group membership")
            require(len(row["dp_members"]) == len(row["cp_members"]) == 2, "Invalid group sizes")
            key = tuple(row["dp_members"])
            groups.setdefault(key, []).append(row)
            require(len(row["local_advantages"]) == len(row["local_masks"]) == row["local_numel"], "Local tensor length mismatch")
            require(all(math.isfinite(x) for x in row["local_advantages"]), "Nonfinite pre-whitening advantages")
            require(all(x == 1 for x in row["local_masks"]), "Primary fixture requires all-one masks")
        require(len(groups) == 2 and all(len(g) == 2 for g in groups.values()), "Incomplete DP groups")
        empty_groups = 0
        for members, rows in groups.items():
            require({r["rank"] for r in rows} == set(members), "DP ledger is not reciprocal")
            require(len({r["cp_rank"] for r in rows}) == 1, "DP group crosses CP coordinates")
            require({r["dp_rank"] for r in rows} == {0, 1}, "DP coordinates invalid")
            counts = [r["local_numel"] for r in rows]
            require(sum(counts) > 1, "Globally empty/degenerate whitening group")
            if min(counts) == 0 < max(counts):
                empty_groups += 1
        require(empty_groups == 1, "Expected exactly one empty/nonempty DP group per role")


def verify_healthy(root):
    require((root / "TRAINING_RETURNED.json").is_file(), "Public train coroutine did not return")
    events = read_events(root)
    entries = grouped(events, "whitening_enter")
    verify_geometry(entries)
    completed, steps = grouped(events, "whitening_complete"), grouped(events, "optimizer_step")
    require(set(completed) == set(entries) == set(steps), "Missing whitening/update evidence")
    for key, row in entries.items():
        role, _ = key
        population = [v for peer in row["dp_members"]
                      for v in entries[(role, peer)]["local_advantages"]]
        mean = sum(population) / len(population)
        variance = sum((x - mean) ** 2 for x in population) / (len(population) - 1)
        expected = [(x - mean) / math.sqrt(variance + 1e-8) for x in row["local_advantages"]]
        actual = completed[key]["normalized"]
        require(len(actual) == len(expected), "Whitening output length changed")
        require(all(math.isclose(x, y, rel_tol=2e-4, abs_tol=2e-5) for x, y in zip(actual, expected)),
                "Whitening differs from independent pooled sample-variance reference")
        step = steps[key]
        require(step["update_successful"] is True and step["grad_norm_finite"] is True
                and step["sampled_finite"] is True, "Optimizer update unsuccessful/nonfinite")
        require(step["rollout_id"] == step["step_id"] == 0, "Unexpected update counter")
    for role in ["actor", "critic"]:
        require(any(row["sampled_changed"] is True for (kind, _), row in steps.items() if kind == role),
                f"No sampled FP32 parameter change demonstrated for {role}")
        inspect_checkpoint(root / role, expected_iteration=0, require_optimizer=True)
    return {"healthy_evidence_passed": True, "actor_optimizer_steps": 1, "critic_optimizer_steps": 1,
            "checkpoint_payloads_present": True, "checkpoint_restore_tested": False,
            "full_online_rollout": False}


def verify_faulty(root):
    events = read_events(root)
    entries = grouped(events, "whitening_enter")
    verify_geometry(entries)
    completions = grouped(events, "whitening_complete")
    calls, ready = grouped(events, "whitening_call"), grouped(events, "advantages_ready")
    log = (root / "stdout.log").read_text(errors="replace")
    watchdog = "Watchdog caught collective operation timeout" in log
    store_timeout = (
        "DistBackendError" in log
        and "store->get" in log
        and "wait timeout after" in log
        and "distributed_c10d.py" in log
        and "all_reduce" in log
    )
    require(watchdog or store_timeout, "No actual target NCCL collective failure evidence")
    divergent = []
    for key, row in entries.items():
        if row["local_numel"] == 0:
            require(key not in completions, "Empty shard did not take the injected skip branch")
            require(key not in calls and key in ready, "No evidence that empty peer advanced past the guard")
            peer = next(rank for rank in row["dp_members"] if rank != row["rank"])
            require(entries[(row["role"], peer)]["local_numel"] > 0, "No nonempty peer")
            require((row["role"], peer) not in completions, "Expected peer completed whitening")
            require((row["role"], peer) in calls, "No proof of nonempty peer entering whitening")
            waiting = entries[(row["role"], peer)]
            timeout_lines = [line for line in log.splitlines()
                             if "Watchdog caught collective operation timeout" in line
                             and re.search(rf"pid={waiting['pid']}(?:[, )])", line)
                             and f"[rank{peer}]" in line]
            if watchdog:
                require(bool(timeout_lines), "Watchdog is not correlated to expected role/PID/rank")
            else:
                # Some NCCL/PyTorch combinations fail while bootstrapping the
                # communicator and report TCPStore wait timeout instead of the
                # watchdog sentence. The call ledger already binds the failure
                # to this peer's target all_reduce; require the complete target
                # stack signature and accept one shared traceback for the group.
                require(store_timeout, "Store timeout is not a target all_reduce failure")
            divergent.append({"role": row["role"], "skipping_rank": row["rank"], "waiting_rank": peer})
    require(len(divergent) == 2, "Need actor and critic divergence")
    require(not (root / "TRAINING_RETURNED.json").exists(), "Faulty training unexpectedly returned")
    return {"faulty_evidence_passed": True, "failure_evidence": "watchdog" if watchdog else "store_wait_timeout",
            "first_divergence": divergent,
            "watchdog_attribution": "first divergence in per-rank ledger; ultimate watchdog may be a later collective"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["healthy", "faulty"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = (verify_healthy if args.kind == "healthy" else verify_faulty)(args.run)
    print(json.dumps(result, indent=2))
