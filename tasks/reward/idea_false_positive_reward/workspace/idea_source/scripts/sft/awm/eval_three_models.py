#!/usr/bin/env python3
"""Evaluate three already-running AWM vLLM services on one shared task sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_TASKS = "/workspace/data/AgentWorldModel-1K/gen_tasks.jsonl"
DEFAULT_AWM_ROOT = "/workspace/awm_source"
DEFAULT_OUTPUT = "/workspace/idea_source/outputs/awm_three_models"
MODELS = (
    ("step_715", (8218, 8219), "global_step_715"),
    ("step_1430", (8220, 8221), "global_step_1430"),
    ("step_2145", (8222, 8223), "global_step_2145"),
)


def read_tasks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for task_idx, task in enumerate(row.get("tasks", [])):
                rows.append({
                    "task_id": f"{row['scenario']}__task_{task_idx}__rollout_0",
                    "scenario": row["scenario"], "task_idx": task_idx,
                    "rollout_idx": 0, "task": task,
                })
    return rows


def read_rollout_stats(path: Path) -> dict:
    records = successes = 0
    statuses = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records += 1
                successes += int(bool(item.get("success")))
                status = str(item.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
    return {
        "records": records,
        "successes": successes,
        "failures": records - successes,
        "pass_rate": successes / records if records else 0.0,
        "statuses": statuses,
    }


def save_progress(output: Path, children: list[tuple[str, subprocess.Popen]], count: int) -> None:
    processes = {name: proc for name, proc in children}
    models = {}
    for name, ports, model in MODELS:
        models[name] = {
            "model": model,
            "ports": list(ports),
            "process_returncode": processes[name].poll(),
            **read_rollout_stats(output / name / "baseline_rollouts.jsonl"),
        }
    progress = {
        "tasks_per_model": count,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models,
    }
    path = output / "progress.json"
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--awm-root", default=DEFAULT_AWM_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--task-timeout", type=float, default=1800.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"manifest_{args.count}_{args.seed}.jsonl"
    all_tasks = read_tasks(Path(args.tasks))
    if args.count < 1 or args.count > len(all_tasks):
        raise ValueError(f"count must be in [1, {len(all_tasks)}]")
    selected = random.Random(args.seed).sample(all_tasks, args.count)
    if not (args.resume and manifest.exists()):
        with manifest.open("w", encoding="utf-8") as stream:
            for item in selected:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    for name, ports, expected_model in MODELS:
        for port in ports:
            url = f"http://127.0.0.1:{port}/v1/models"
            started = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=args.health_timeout) as response:
                    payload = json.load(response)
                served_models = {item.get("id") for item in payload.get("data", [])}
            except Exception as exc:
                raise RuntimeError(f"Unhealthy model endpoint {url}: {type(exc).__name__}: {exc}") from exc
            if expected_model not in served_models:
                raise RuntimeError(f"Endpoint {url} serves {sorted(served_models)}, expected {expected_model}")
            print(f"healthy {name} endpoint={url} latency={time.monotonic() - started:.2f}s", flush=True)

    runner = Path(__file__).resolve().parents[2] / "awm" / "run_awm_rollouts.py"
    children: list[tuple[str, subprocess.Popen]] = []
    for name, ports, model in MODELS:
        model_dir = output / name
        model_dir.mkdir(parents=True, exist_ok=True)
        log = (output / f"{name}.log").open("w", encoding="utf-8")
        command = [
            sys.executable, str(runner), "--awm-root", args.awm_root,
            "--data-root", str(Path(args.tasks).parent), "--task-manifest", str(manifest),
            "--output-dir", str(model_dir),
            "--endpoints", ",".join(f"http://127.0.0.1:{port}/v1" for port in ports),
            "--model-names", model,
            "--per-endpoint-concurrency", str(args.concurrency),
            "--max-iterations", str(args.max_iterations), "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--task-timeout", str(args.task_timeout),
            "--resume" if args.resume else "--no-resume",
        ]
        proc = subprocess.Popen(command, cwd=args.awm_root, stdout=log, stderr=subprocess.STDOUT)
        children.append((name, proc))
        print(f"started {name} pid={proc.pid} ports={ports}", flush=True)

    while any(proc.poll() is None for _, proc in children):
        save_progress(output, children, args.count)
        time.sleep(10)
    save_progress(output, children, args.count)
    exit_codes = {name: proc.returncode for name, proc in children}
    comparison = []
    for name, ports, model in MODELS:
        summary_path = output / name / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        comparison.append({
            "name": name, "model": model, "ports": list(ports),
            "process_returncode": exit_codes[name], "tasks": args.count,
            "records": summary.get("records", 0), "successes": summary.get("successes", 0),
            "failures": summary.get("failures", 0), "pass_rate": summary.get("pass_rate", 0.0),
            "output_dir": str(output / name),
        })
    result = {
        "manifest": str(manifest), "count": args.count, "seed": args.seed,
        "concurrency_per_endpoint": args.concurrency,
        "concurrency_per_model": args.concurrency * 2,
        "comparison": comparison,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
