#!/usr/bin/env python3
"""Run a small, reproducible multi-turn smoke test against three AWM services.

The test samples three AWM tasks and sends each task as a conversation with
three user turns.  It uses the same MCP-oriented system prompt as the AWM
agent, but does not start an environment or execute tools; this isolates the
model-service conversation behavior and keeps the test safe to run repeatedly.
Every request, response, and service error is written to a JSONL transcript.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


DEFAULT_TASKS = Path("/workspace/data/AgentWorldModel-1K/gen_tasks.jsonl")
DEFAULT_OUTPUT = Path("tests/results/awm_three_services_multiturn.jsonl")
BASE_URLS = {
    "model_8220": "http://127.0.0.1:8220/v1",
    "model_8221": "http://127.0.0.1:8221/v1",
    "model_8222": "http://127.0.0.1:8222/v1",
}

SYSTEM_PROMPT = (
    "You are an Agent World Model agent operating in an MCP environment. "
    "Reason carefully about the user's task. If tools were available, you "
    "would use list_tools first and then call_tool; in this smoke test, only "
    "produce the next action or a concise final answer."
)


def load_tasks(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("tasks"):
                    records.append(record)
    rng = random.Random(seed)
    rng.shuffle(records)
    selected = records[:count]
    if len(selected) < count:
        raise ValueError(f"only found {len(selected)} task records in {path}")
    return selected


def served_model(base_url: str) -> tuple[str | None, str | None]:
    try:
        with httpx.Client(base_url=base_url, timeout=20.0, trust_env=False) as http:
            response = http.get("/models")
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                return None, "models endpoint returned no models"
            return data[0].get("id"), None
    except Exception as exc:  # keep one broken service from aborting the run
        return None, f"{type(exc).__name__}: {exc}"


def run_conversation(base_url: str, model: str, task: str, max_tokens: int) -> dict[str, Any]:
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=90.0,
        max_retries=0,
        http_client=httpx.Client(timeout=90.0, trust_env=False),
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns = [
        task,
        "检查你刚才的方案：指出可能遗漏的步骤、参数或风险，并给出修正版下一步。",
        "现在给出最终的简洁结果；如果无法实际调用工具，请明确说明需要什么工具或环境。",
    ]
    output_turns: list[dict[str, Any]] = []
    for turn_index, user_content in enumerate(turns, start=1):
        messages.append({"role": "user", "content": user_content})
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            content = message.content or ""
            error = None
            messages.append({"role": "assistant", "content": content})
        except Exception as exc:
            content = ""
            error = f"{type(exc).__name__}: {exc}"
        output_turns.append(
            {
                "turn": turn_index,
                "user": user_content,
                "assistant": content,
                "error": error,
                "latency_s": round(time.monotonic() - started, 3),
                "assistant_nonempty": bool(content.strip()),
            }
        )
        if error:
            break
    return {"turns": output_turns, "nonempty_turns": sum(t["assistant_nonempty"] for t in output_turns)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    selected = load_tasks(args.tasks, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for service, base_url in BASE_URLS.items():
        model, discovery_error = served_model(base_url)
        for sample_index, record in enumerate(selected):
            task = record["tasks"][sample_index % len(record["tasks"])]
            row: dict[str, Any] = {
                "run_started": run_started,
                "service": service,
                "base_url": base_url,
                "model": model,
                "sample_index": sample_index,
                "scenario": record.get("scenario"),
                "task": task,
                "discovery_error": discovery_error,
            }
            if model and not discovery_error:
                row.update(run_conversation(base_url, model, task, args.max_tokens))
            else:
                row.update({"turns": [], "nonempty_turns": 0})
            rows.append(row)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(args.output),
        "services": len(BASE_URLS),
        "samples": len(selected),
        "rows": len(rows),
        "nonempty_turns": sum(row["nonempty_turns"] for row in rows),
        "total_turns": sum(len(row["turns"]) for row in rows),
        "service_status": {
            row["service"]: {"model": row["model"], "discovery_error": row["discovery_error"]}
            for row in rows
            if row["sample_index"] == 0
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
