#!/usr/bin/env python3
"""Run the same task manifest against every checkpoint and compare pass rates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STEPS = (100, 300, 500, 700, 900, 1100, 1300, 1600)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--awm-root", default="/workspace/awm_source")
    p.add_argument("--data-root", default="/workspace/data/AgentWorldModel-1K")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--base-port", type=int, default=8120)
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--per-model-concurrency", type=int, default=32)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_awm_rollouts.py")
    python = Path(args.awm_root) / ".venv/bin/python"
    children: list[tuple[int, subprocess.Popen[str]]] = []

    for index, step in enumerate(STEPS):
        model = f"qwen3-8b-step-{step}"
        output_dir = root / f"step_{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = root / f"step_{step}.runner.log"
        command = [
            str(python), str(runner),
            "--awm-root", args.awm_root,
            "--data-root", args.data_root,
            "--task-manifest", args.manifest,
            "--output-dir", str(output_dir),
            "--base-port", str(args.base_port + index),
            "--num-endpoints", "1",
            "--model-names", model,
            "--per-endpoint-concurrency", str(args.per_model_concurrency),
            "--max-iterations", str(args.max_iterations),
            "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--resume" if args.resume else "--no-resume",
        ]
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=args.awm_root, stdout=log, stderr=subprocess.STDOUT, text=True)
        children.append((step, proc))
        print(f"started step={step} pid={proc.pid} port={args.base_port + index} output={output_dir}", flush=True)

    exit_codes = {}
    for step, proc in children:
        exit_codes[step] = proc.wait()
        print(f"finished step={step} returncode={exit_codes[step]}", flush=True)

    comparison = []
    for index, step in enumerate(STEPS):
        output_dir = root / f"step_{step}"
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            try:
                item = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                item = {}
        else:
            item = {}
        comparison.append({
            "step": step,
            "model": f"qwen3-8b-step-{step}",
            "output_dir": str(output_dir),
            "port": args.base_port + index,
            "process_returncode": exit_codes[step],
            "tasks": item.get("tasks", 0),
            "records": item.get("records", 0),
            "successes": item.get("successes", 0),
            "failures": item.get("failures", 0),
            "pass_rate": item.get("pass_rate", 0.0),
            "status": "complete" if exit_codes[step] == 0 and item.get("records", 0) == item.get("tasks", 0) else "incomplete",
        })

    result = {
        "task_manifest": str(Path(args.manifest).resolve()),
        "task_count": 100,
        "steps": list(STEPS),
        "max_tokens": args.max_tokens,
        "max_iterations": args.max_iterations,
        "per_model_concurrency": args.per_model_concurrency,
        "comparison": comparison,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
