#!/usr/bin/env python3
"""Call the AWM SQL Judge on real trajectories saved by a rollout.

The rollout JSONL contains the model's actions and episode score, but does not
contain the original SQL-verifier payload.  Therefore this probe replays the
real trajectory and passes the recorded score as a small verifier-result
summary.  It is intended as an interactive smoke test for the Judge endpoint,
not as a replacement for the database verifier.

Example:
  python tests/test_awm_sql_judge_from_rollouts.py --count 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

AWM_ROOT = Path(os.environ.get("AWM_ROOT", "/workspace/awm_source"))
sys.path.insert(0, str(AWM_ROOT))

from awm.core.verify import run_llm_judge  # noqa: E402


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _estimate_tokens(task: str, trajectory: list[dict[str, Any]], verifier_result: dict[str, Any], result: Any) -> tuple[int, int, int]:
    """Estimate tokens when the wrapper does not expose OpenAI ``usage``."""
    task_payload = {
        "user_task": task,
        "actual_execution_steps": len(trajectory),
        "trajectory": trajectory,
    }
    verification_json = {"code_execution_result": verifier_result}
    user_text = (
        "task_json:\n" + json.dumps(task_payload, ensure_ascii=False, indent=2, default=str)
        + "\n\nverification_json:\n"
        + json.dumps(verification_json, ensure_ascii=False, indent=2, default=str)
    )
    # The fixed system prompt in awm.core.verify is roughly this many chars;
    # 4 chars/token is a conservative mixed English/JSON approximation.
    system_tokens = 1900 // 4
    user_tokens = max(1, len(user_text) // 4)
    output_tokens = max(1, len(json.dumps(result, ensure_ascii=False, default=str)) // 4)
    return system_tokens, user_tokens, output_tokens


def _task_from_input(text: str) -> str:
    marker = "User task:\n"
    if marker in text:
        return text.split(marker, 1)[1].split("\nassistant", 1)[0].strip()
    return text.strip()


def _trajectory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        output = str(row.get("output", ""))
        match = TOOL_CALL_RE.search(output)
        tool_calls = []
        if match:
            try:
                tool_calls = [json.loads(match.group(1).strip())]
            except json.JSONDecodeError:
                tool_calls = []
        result.append(
            {
                "iteration": int(row.get("step_num", len(result))) + 1,
                "role": "assistant",
                "content": output,
                "tool_calls": tool_calls,
                # obs_text is the real environment observation saved for this
                # step; it is the closest replayable equivalent of bridge's
                # tool_response because rollout JSONL omits that field.
                "tool_response": {"content": str(row.get("obs_text", ""))},
            }
        )
    return result


def _is_completed(rows: list[dict[str, Any]], max_steps: int) -> bool:
    last = rows[-1]
    if int(last.get("step_num", -1)) + 1 >= max_steps:
        return True
    output = str(last.get("output", "")).strip()
    return bool(output) and "<tool_call>" not in output


def load_examples(path: Path, count: int, max_steps: int) -> tuple[list[list[dict[str, Any]]], int]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(str(row["traj_uid"]), []).append(row)
    normalized = []
    for rows in grouped.values():
        by_step = OrderedDict((int(row.get("step_num", i)), row) for i, row in enumerate(rows))
        normalized.append(list(by_step.values()))
    examples = [rows for rows in normalized if _is_completed(rows, max_steps)]
    skipped = len(normalized) - len(examples)
    # Prefer a mix of successful and unsuccessful real trajectories.
    examples.sort(key=lambda rows: (-float(rows[-1].get("score", 0.0)), len(rows)))
    if count <= len(examples):
        selected = examples[:count]
        if count >= 2:
            failed = next((x for x in examples if float(x[-1].get("score", 0.0)) < 1.0), None)
            if failed is not None:
                selected[-1] = failed
        return selected, skipped
    return examples, skipped


async def main(args: argparse.Namespace) -> int:
    rows_by_traj, skipped = load_examples(Path(args.rollouts), args.count, args.max_steps)
    base_url = args.base_url or os.environ.get(
        "AWM_JUDGE_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
    )
    model = args.model or os.environ.get("AWM_JUDGE_MODEL", "glm-5.3-flash")
    api_key = args.api_key or os.environ.get("AWM_JUDGE_API_KEY", os.environ.get("GLM_API_KEY", ""))

    print(f"rollouts={args.rollouts} completed_examples={len(rows_by_traj)} skipped_incomplete={skipped} judge={base_url} model={model}")
    for index, rows in enumerate(rows_by_traj, 1):
        task = _task_from_input(str(rows[0].get("input", "")))
        score = float(rows[-1].get("score", 0.0))
        trajectory = _trajectory(rows)
        verifier_result = {
            "replay_rollout_score": score,
            "recorded_steps": len(rows),
            "note": "The JSONL has no SQL verifier payload; this is a replay smoke test.",
        }
        classification, result = await run_llm_judge(
            task=task,
            verifier_result=verifier_result,
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=model,
            trajectory=trajectory,
        )
        print(f"\n=== example {index} traj_uid={rows[0]['traj_uid']} recorded_score={score} ===")
        print(f"task: {task}")
        print(f"judge_classification: {classification}")
        # run_llm_judge returns the parsed Judge JSON, not the provider usage.
        # Report deterministic prompt/output token estimates for this probe.
        system_tokens, user_tokens, output_tokens = _estimate_tokens(task, trajectory, verifier_result, result)
        print(f"input_tokens_estimate: {system_tokens + user_tokens} (system={system_tokens}, user={user_tokens})")
        print(f"output_tokens_estimate: {output_tokens}")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        default="outputs/awm_opsd_qwen3_8b_step693_wandb_v0/rollouts/10.jsonl",
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
