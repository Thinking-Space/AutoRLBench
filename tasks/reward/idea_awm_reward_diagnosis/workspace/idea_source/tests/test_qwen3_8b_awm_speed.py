#!/usr/bin/env python3
"""Latency smoke test for the eight locally deployed AWM Qwen3 checkpoints.

Each endpoint receives the same two prompts: a short ordinary prompt and one
task sampled from the shared AWM manifest.  The complete model output and
request timing are printed for every checkpoint.  Streaming is used so both
time-to-first-nonempty-token and end-to-end latency are visible.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx
from openai import OpenAI


DEFAULT_STEPS = (100, 300, 500, 700, 900, 1100, 1300, 1600)
DEFAULT_MANIFEST = Path(
    "/workspace/idea_source/outputs/awm_qwen3_8b_eval/task_manifest.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-port", type=int, default=8120)
    p.add_argument(
        "--steps",
        default=",".join(map(str, DEFAULT_STEPS)),
        help="Comma-separated checkpoint steps to test, for example: 100,1600.",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request Qwen thinking mode (default reproduces the AWM agent setting).",
    )
    p.add_argument(
        "--ordinary-prompt",
        default="Reply with exactly: speed smoke test passed.",
    )
    return p.parse_args()


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_case(
    client: OpenAI,
    model: str,
    label: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    first_nonempty = None
    chunks: list[str] = []
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=True,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        },
    )
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        text = getattr(delta, "content", None) or ""
        if text:
            if first_nonempty is None:
                first_nonempty = time.perf_counter()
            chunks.append(text)
    finished = time.perf_counter()
    output = "".join(chunks)
    usage = getattr(response, "usage", None)
    result = {
        "case": label,
        "model": model,
        "prompt_chars": len(prompt),
        "output_chars": len(output),
        "time_to_first_nonempty_sec": None
        if first_nonempty is None
        else round(first_nonempty - started, 3),
        "total_time_sec": round(finished - started, 3),
        "usage": str(usage) if usage else None,
        "output": output,
    }
    print(f"\n[{model}] {label}")
    print(
        f"time_to_first_nonempty={result['time_to_first_nonempty_sec']}s "
        f"total={result['total_time_sec']}s output_chars={len(output)}"
    )
    print("--- model output ---")
    print(output)
    print("--- end output ---")
    return result


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.manifest)
    if not tasks:
        raise SystemExit(f"No tasks found in {args.manifest}")
    if args.task_index is None:
        task_index = random.Random(args.seed).randrange(len(tasks))
    else:
        task_index = args.task_index
    if not 0 <= task_index < len(tasks):
        raise SystemExit(f"--task-index must be in [0, {len(tasks)})")
    task = tasks[task_index]
    dataset_prompt = task["task"]
    print(f"Dataset task index={task_index}: {dataset_prompt}")
    print(
        f"max_tokens={args.max_tokens} temperature={args.temperature} "
        f"enable_thinking={args.enable_thinking}"
    )

    steps = [int(value.strip()) for value in args.steps.split(",") if value.strip()]
    invalid = [step for step in steps if step not in DEFAULT_STEPS]
    if invalid:
        raise SystemExit(f"Unknown steps: {invalid}; expected values from {DEFAULT_STEPS}")

    results: list[dict] = []
    for step in steps:
        port = args.base_port + DEFAULT_STEPS.index(step)
        base_url = f"http://127.0.0.1:{port}/v1"
        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=args.timeout,
            http_client=httpx.Client(trust_env=False, timeout=args.timeout),
        )
        model = f"qwen3-8b-step-{step}"
        for label, prompt in (("ordinary", args.ordinary_prompt), ("dataset", dataset_prompt)):
            try:
                results.append(run_case(client, model, label, prompt, args))
            except Exception as exc:
                result = {
                    "case": label,
                    "model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                results.append(result)
                print(f"\n[{model}] {label} ERROR: {result['error']}")

    print("\n=== speed summary ===")
    for result in results:
        if "error" in result:
            print(f"{result['model']}\t{result['case']}\tERROR\t{result['error']}")
        else:
            print(
                f"{result['model']}\t{result['case']}\t"
                f"first={result['time_to_first_nonempty_sec']}s\t"
                f"total={result['total_time_sec']}s\tchars={result['output_chars']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
