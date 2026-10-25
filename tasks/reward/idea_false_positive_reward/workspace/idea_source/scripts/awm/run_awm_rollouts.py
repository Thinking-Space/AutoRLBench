#!/usr/bin/env python3
"""Run AWM MCP rollouts and emit SEED-compatible JSONL records.

Each task gets an isolated AWM SQLite/MCP environment.  Work is split over
OpenAI-compatible endpoints, with at most ``per_endpoint_concurrency`` agent
processes per endpoint.  Results are appended as they finish, so restarting
the command resumes from completed task ids.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


# ProcessPool workers do not reliably inherit terminal signal semantics. Keep
# track of the child currently owned by each worker so shutdown can remove the
# whole agent process group instead of leaving it orphaned.
_MAIN_PID = os.getpid()
_ACTIVE_CHILD: subprocess.Popen | None = None
_POOL: concurrent.futures.ProcessPoolExecutor | None = None


def _terminate_active_child() -> None:
    child = _ACTIVE_CHILD
    if child is None or child.poll() is not None:
        return
    _terminate_process_group(child)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Stop a command and its descendants, escalating if it ignores SIGTERM."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _handle_shutdown(signum: int, _frame: Any) -> None:
    """Propagate shutdown to pool workers and their active agent child."""
    global _POOL
    owner_pid = int(os.environ.get("_SEED_AWM_ROLLOUT_MAIN_PID", _MAIN_PID))
    if os.getpid() == owner_pid:
        pool = _POOL
        if pool is not None:
            for worker in pool._processes.values():  # stdlib has no public terminate API
                try:
                    os.kill(worker.pid, signum)
                except ProcessLookupError:
                    pass
        raise KeyboardInterrupt
    _terminate_active_child()
    raise SystemExit(128 + signum)


def _install_worker_signal_handlers() -> None:
    """Install cleanup handlers in workers for spawn/forkserver start methods."""
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--awm-root", default="/workspace/awm_source")
    p.add_argument("--data-root", default="/workspace/data/AgentWorldModel-1K")
    p.add_argument("--output-dir", default="/workspace/idea_source/outputs/awm_qwen35_rollouts")
    p.add_argument("--task-manifest", default=None, help="JSONL task manifest shared by all model endpoints")
    p.add_argument("--base-port", type=int, default=8020)
    p.add_argument("--num-endpoints", type=int, default=8)
    p.add_argument("--endpoints", default=None, help="Comma-separated API base URLs; overrides base-port/num-endpoints")
    p.add_argument("--model-names", default="qwen35-9b", help="Comma-separated served model names, one per endpoint")
    p.add_argument("--per-endpoint-concurrency", type=int, default=16)
    p.add_argument("--rollouts-per-task", type=int, default=1)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--task-timeout", type=float, default=600.0)
    p.add_argument("--verify-timeout", type=float, default=120.0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log, timeout: float) -> tuple[int, bool]:
    """Run a command with a hard timeout and clean up its whole process group."""
    global _ACTIVE_CHILD
    proc = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    _ACTIVE_CHILD = proc
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        return 124, True
    except BaseException:
        _terminate_process_group(proc)
        raise
    finally:
        if _ACTIVE_CHILD is proc:
            _ACTIVE_CHILD = None


def stable_id(scenario: str, task_idx: int, rollout_idx: int) -> str:
    return f"{scenario}__task_{task_idx}__rollout_{rollout_idx}"


def make_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_manifest:
        rows = read_jsonl(Path(args.task_manifest))
        tasks = []
        for row in rows:
            tasks.append({
                "task_id": row["task_id"],
                "scenario": row["scenario"],
                "task_idx": int(row["task_idx"]),
                "rollout_idx": int(row.get("rollout_idx", 0)),
                "task": row["task"],
            })
        if args.max_tasks is not None:
            tasks = tasks[: max(0, args.max_tasks)]
        return tasks
    rows = read_jsonl(Path(args.data_root) / "gen_tasks.jsonl")
    tasks: list[dict[str, Any]] = []
    for row in rows:
        scenario = row["scenario"]
        for task_idx, task in enumerate(row.get("tasks", [])):
            for rollout_idx in range(args.rollouts_per_task):
                tasks.append({
                    "task_id": stable_id(scenario, task_idx, rollout_idx),
                    "scenario": scenario,
                    "task_idx": task_idx,
                    "rollout_idx": rollout_idx,
                    "task": task,
                })
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]
    return tasks


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("task_id"):
                    latest[str(item["task_id"])] = str(item.get("status", ""))
            except json.JSONDecodeError:
                continue
    # Any persisted record counts as handled.  This lets callers resume only
    # tasks with no output at all, including when error records are intentional.
    return set(latest)


def awm_subprocess_env(awm_root: Path) -> dict[str, str]:
    """Use a small clean environment; nested tmux conda state breaks mcp-agent."""
    env = {}
    for key in ("HOME", "USER", "LANG", "TERM", "SHELL", "TMPDIR", "TMP", "TEMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    env["PATH"] = f"{awm_root / '.venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin"
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def run_one(item: dict[str, Any], endpoint: str, model_name: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    task_id = item["task_id"]
    run_dir = output_dir / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    awm = Path(args.awm_root)
    data = Path(args.data_root)
    agent_cmd = [
        str(awm / ".venv/bin/awm"), "agent",
        "--scenario", item["scenario"], "--task_id", str(item["task_idx"]),
        "--envs_path", str(data / "gen_envs.jsonl"),
        "--tasks_path", str(data / "gen_tasks.jsonl"),
        "--db_path", str(data / "gen_db.jsonl"),
        "--sample_path", str(data / "gen_sample.jsonl"),
        "--api_url", endpoint, "--model", model_name,
        "--output_dir", str(run_dir),
        "--max_iterations", str(args.max_iterations),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
    ]
    log_path = run_dir / "agent.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        agent_returncode, agent_timed_out = run_logged(
            agent_cmd, cwd=awm, env=awm_subprocess_env(awm), log=log,
            timeout=args.task_timeout,
        )
    result: dict[str, Any] = {
        **item, "endpoint": endpoint, "model": model_name, "agent_returncode": agent_returncode,
        "run_dir": str(run_dir), "elapsed_sec": round(time.time() - started, 3),
    }
    trajectory_path = run_dir / "trajectory.json"
    if agent_returncode != 0 or not trajectory_path.exists():
        status = "agent_timeout" if agent_timed_out else "agent_error"
        result.update({"success": False, "reward": 0.0, "status": status})
        return result

    verify_cmd = [
        str(awm / ".venv/bin/awm"), "verify", "--input", str(run_dir), "--mode", "code",
        "--verifier_code_path", str(data / "gen_verifier.pure_code.jsonl"),
    ]
    with (run_dir / "verify.log").open("w", encoding="utf-8") as log:
        verify_returncode, verify_timed_out = run_logged(
            verify_cmd, cwd=awm, env=awm_subprocess_env(awm), log=log,
            timeout=args.verify_timeout,
        )
    verify_path = run_dir / "verify.code.json"
    verify_data: dict[str, Any] = {}
    if verify_path.exists():
        try:
            verify_data = json.loads(verify_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    reward_type = verify_data.get("reward_type", "judge_error")
    success = reward_type == "complete"
    result.update({
        "success": success, "reward": 1.0 if success else 0.0,
        "status": reward_type if verify_returncode == 0 else ("verify_timeout" if verify_timed_out else "verify_error"),
        "verify": verify_data.get("verify_result", {}),
    })
    # This is the format consumed by scripts/sft/checklist/run_skill_pipeline.py.
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        messages = trajectory.get("messages", [])
        tool_names: set[str] = set()
        for message in messages:
            for call in message.get("tool_calls", []) or []:
                args_obj = call.get("arguments", {}) if isinstance(call, dict) else {}
                if isinstance(args_obj, str):
                    try:
                        args_obj = json.loads(args_obj)
                    except json.JSONDecodeError:
                        args_obj = {}
                if isinstance(args_obj, dict) and args_obj.get("tool_name"):
                    tool_names.add(str(args_obj["tool_name"]))
                elif isinstance(call, dict) and call.get("name") not in {"list_tools", "call_tool"}:
                    tool_names.add(str(call["name"]))
            content = message.get("content", "")
            if isinstance(content, str) and content.startswith("Available MCP Tools"):
                import re
                tool_names.update(re.findall(r"^\d+\. (mcp_tool_[A-Za-z0-9_]+)", content, re.MULTILINE))
        result.update({
            "task_description": trajectory.get("task", item["task"]),
            "tools": [{"name": name} for name in sorted(tool_names)], "messages": messages,
            "trajectory": trajectory.get("trajectory", []),
            "scenario": trajectory.get("scenario", item["scenario"]),
            "task_idx": trajectory.get("task_id", item["task_idx"]),
            "source_uuid": task_id,
        })
    except Exception as exc:
        result["conversion_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_rollouts.jsonl"
    tasks = make_tasks(args)
    completed = load_completed(output_path) if args.resume else set()
    pending = [t for t in tasks if t["task_id"] not in completed]
    endpoints = (
        [value.strip().rstrip("/") for value in args.endpoints.split(",") if value.strip()]
        if args.endpoints
        else [f"http://127.0.0.1:{args.base_port + i}/v1" for i in range(args.num_endpoints)]
    )
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    model_names = [name.strip() for name in args.model_names.split(",") if name.strip()]
    if len(model_names) == 1:
        model_names *= len(endpoints)
    if len(model_names) != len(endpoints):
        raise ValueError(f"Expected 1 or {len(endpoints)} model names, got {len(model_names)}")
    print(f"tasks={len(tasks)} pending={len(pending)} endpoints={len(endpoints)} concurrency={args.per_endpoint_concurrency}", flush=True)
    completed_count = 0
    current_successes = 0
    current_statuses: dict[str, int] = {}
    # AWM's MCP client creates event loops and subprocesses.  Isolate every
    # rollout in a process rather than a thread to avoid shared runtime state.
    worker_count = min(len(pending), len(endpoints) * args.per_endpoint_concurrency)
    global _POOL
    # Preserve the owner identity when multiprocessing uses spawn/forkserver,
    # where this module is imported afresh inside each worker.
    os.environ["_SEED_AWM_ROLLOUT_MAIN_PID"] = str(_MAIN_PID)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, worker_count), initializer=_install_worker_signal_handlers,
    ) as pool:
        _POOL = pool
        # Submit round-robin so the first worker wave uses all endpoints;
        # grouped buckets would fill only endpoint 0 for a long time.
        futures = {
            pool.submit(
                run_one,
                item,
                endpoints[index % len(endpoints)],
                model_names[index % len(endpoints)],
                args,
                output_dir,
            ): item
            for index, item in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    **item,
                    "success": False,
                    "reward": 0.0,
                    "status": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(output_path, record, threading.Lock())
            completed_count += 1
            current_successes += int(bool(record.get("success")))
            status = str(record.get("status", "unknown"))
            current_statuses[status] = current_statuses.get(status, 0) + 1
            if completed_count % 10 == 0 or completed_count == len(pending):
                progress = {
                    "total_tasks": len(tasks),
                    "previously_completed": len(completed),
                    "completed_this_run": completed_count,
                    "remaining": len(pending) - completed_count,
                    "successes_this_run": current_successes,
                    "pass_rate_this_run": current_successes / completed_count,
                    "statuses_this_run": current_statuses,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                progress_path = output_dir / "progress.json"
                temp_progress_path = progress_path.with_suffix(".json.tmp")
                temp_progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                temp_progress_path.replace(progress_path)
                print(f"progress={completed_count}/{len(pending)} task_id={record['task_id']} status={record.get('status')}", flush=True)
    _POOL = None

    records = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    success = sum(bool(item.get("success")) for item in records)
    summary = {
        "tasks": len(tasks),
        "records": len(records),
        "successes": success,
        "failures": len(records) - success,
        "pass_rate": success / len(records) if records else 0.0,
        "base_port": args.base_port,
        "model_names": model_names,
        "max_tokens": args.max_tokens,
        "per_endpoint_concurrency": args.per_endpoint_concurrency,
        "task_manifest": args.task_manifest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
