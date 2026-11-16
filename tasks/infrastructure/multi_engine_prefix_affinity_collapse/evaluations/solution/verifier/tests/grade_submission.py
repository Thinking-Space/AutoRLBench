"""Grade the system rerun of a repaired workspace.

Reads only files written during the fresh run by code that is byte-identical to the pristine copies
under resources/pristine/, checks that the experiment contract was kept, and compares the task metric
with the calibrated threshold in resources/grader_spec.json.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import sys
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1]


def read_jsonl(paths):
    rows = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def runner_integrity(workspace: Path, spec: dict):
    details = {}
    for rel in spec["pinned_files"]:
        pristine = VERIFIER / "resources" / "pristine" / rel
        staged = workspace / rel
        try:
            details[rel] = staged.read_bytes() == pristine.read_bytes()
        except OSError:
            details[rel] = False
    return all(details.values()), details


def contract(cfg: dict, spec: dict):
    pinned = json.loads((VERIFIER / "resources" / "pristine" / "scripts" / "run_config.json").read_text())
    violations = []
    for key in sorted(set(pinned) | set(cfg)):
        if key in spec["free_config_sections"]:
            continue
        if key == "env":
            free = tuple(spec.get("free_env_prefixes", []))
            mine = {k: v for k, v in cfg.get("env", {}).items() if not k.startswith(free)}
            ref = {k: v for k, v in pinned.get("env", {}).items() if not k.startswith(free)}
            if mine != ref:
                violations.append("env")
        elif cfg.get(key) != pinned.get(key):
            violations.append(key)
    return violations


def training_completed(run: Path, spec: dict):
    rollouts = sorted(run.glob("rollout.*.json"))
    updates = []
    for rollout_id in range(spec["rollouts"]):
        path = run / "optimizer" / f"step.{rollout_id}.0.rank0.json"
        if path.exists():
            record = json.loads(path.read_text())
            updates.append(bool(record.get("update_successful")) and (record.get("grad_norm") or 0) > 0
                           and record.get("called_from_trainer") is True)
        else:
            updates.append(False)
    return len(rollouts) >= spec["rollouts"] and all(updates), {"rollouts": len(rollouts), "updates": updates}


def routing_metric(run: Path, spec: dict):
    turns = read_jsonl(sorted((run / "turns").glob("*.jsonl*")))
    by_session = collections.defaultdict(dict)
    for row in turns:
        if row.get("event") == "generate":
            by_session[row["session"]][row["turn"]] = row
    reused = total = 0
    for session_turns in by_session.values():
        for turn, row in session_turns.items():
            previous = session_turns.get(turn - 1)
            if turn == 0 or previous is None:
                continue
            total += 1
            reused += (row.get("cached_tokens") or 0) >= 0.9 * (previous.get("prompt_tokens") or 0)
    share = reused / total if total else 0.0
    sane = len(by_session) >= spec["min_sessions"] and total >= spec["min_follow_up_turns"]
    return share, sane, {"sessions": len(by_session), "follow_up_turns": total, "reusing_previous_prompt": reused}


def sandbox_metric(run: Path, spec: dict):
    walls = {}
    for rollout_id in range(spec["metric_rollouts"]):
        record = json.loads((run / f"rollout.{rollout_id}.json").read_text())
        walls[rollout_id] = float(record["wall_s"])
    value = sum(walls.values())
    access = read_jsonl(sorted(run.glob("sandbox_access.jsonl*")))
    turns = read_jsonl(sorted((run / "turns").glob("*.jsonl*")))
    ops = collections.Counter(row["op"] for row in access)
    per_sandbox = collections.defaultdict(set)
    for row in access:
        if "sandbox" in row:
            per_sandbox[row["sandbox"]].add(row["op"])
    # Sandboxes still mid-setup when the producer stops are not evidence of skipped setup, so the
    # check applies to those that actually served a tool call.
    used = [seen for seen in per_sandbox.values() if "exec" in seen]
    complete_setups = sum({"create", "upload", "install"} <= seen for seen in used)
    tool_calls = sum(len(row.get("calls", [])) for row in turns if row.get("event") == "tools")
    sessions = {row["session"] for row in turns if row.get("event") == "generate"}
    env_errors = sum(1 for row in turns if row.get("event") == "env_error")
    # Trajectories still in flight when the producer stops never reach their close, so deletes
    # track *finished* trajectories, not creates.
    finished = sum(1 for row in turns if row.get("event") in ("done", "env_error", "engine_abort"))
    checks = {
        "every_sandbox_fully_set_up": complete_setups >= 0.98 * len(used),
        "one_sandbox_per_session": ops["create"] >= 0.98 * len(sessions),
        "tool_calls_executed_in_sandboxes": ops["exec"] >= 0.95 * tool_calls,
        "sandboxes_deleted": ops["delete"] >= 0.95 * finished,
        "environment_errors_bounded": env_errors <= spec["max_env_error_fraction"] * max(1, len(sessions)),
    }
    details = {"rollout_wall_s": walls, "ops": dict(ops), "complete_setups": complete_setups, "sandboxes_used": len(used),
               "tool_calls": tool_calls, "sessions": len(sessions), "finished_trajectories": finished,
               "env_errors": env_errors, "checks": checks}
    return value, all(checks.values()), details


def main() -> int:
    results = Path(os.environ["AUTORLBENCH_RESULTS"])
    workspace = Path(os.environ.get("AUTORLBENCH_WORKSPACE", "/workspace"))
    spec = json.loads((VERIFIER / "resources" / "grader_spec.json").read_text())
    runs = sorted(p for p in (results / "experiment_logs").glob("*") if (p / "launch.json").is_file())
    outcome = {"passed": False, "score": 0.0, "metrics": {}}
    if not runs:
        outcome["metrics"]["failure"] = "no run under experiment_logs"
    else:
        run = runs[-1]
        integrity_ok, integrity = runner_integrity(workspace, spec)
        cfg = json.loads((run / "launch.json").read_text())["config"]
        violations = contract(cfg, spec)
        completed, completion = training_completed(run, spec)
        if spec["case"] == "routing":
            value, sane, details = routing_metric(run, spec)
            metric_ok = value >= spec["threshold"]
        else:
            value, sane, details = sandbox_metric(run, spec)
            metric_ok = value <= spec["threshold"]
        passed = integrity_ok and not violations and completed and sane and metric_ok
        outcome = {"passed": passed, "score": float(passed), "metrics": {
            "metric_key": spec["metric_key"], "metric_value": value, "threshold": spec["threshold"],
            "runner_integrity": integrity_ok, "runner_integrity_files": integrity,
            "contract_violations": violations, "training_completed": completed, **completion,
            "workload_intact": sane, **details}}
    (results / "result.json").write_text(json.dumps(outcome, indent=2, default=str) + "\n")
    (results / "metrics.json").write_text(json.dumps(outcome["metrics"], indent=2, default=str) + "\n")
    print(json.dumps(outcome, default=str))
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
