#!/usr/bin/env python3
"""End-to-end test for the AWM Qwen3 SFT checkpoint served by vLLM.

The test uses a trajectory-shaped prompt and checks that the model returns the
same JSON skill schema used by ``candidate_skills.jsonl``.  It talks to an
OpenAI-compatible vLLM endpoint, so it can be run against a local server or a
remote endpoint with ``--base-url``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


DEFAULT_SKILLS = Path(
    "/workspace/idea_source/outputs/awm_qwen38_skills/candidate_skills.jsonl"
)
DEFAULT_MODEL = "/local/user/models/qwen3_8b_awm_skill_sft_verl/global_step_1600_vllm"


def load_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        item = json.loads(stream.readline())
    required = {"episode_summary", "episode_skill", "step_skills"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"reference skill is missing fields: {sorted(missing)}")
    return item


def build_trajectory(reference: dict[str, Any]) -> str:
    """Create a realistic trajectory from the reference task semantics."""
    return json.dumps(
        {
            "task": "Analyze 50 customer reviews for sentiment and create a batch analysis task.",
            "messages": [
                {
                    "role": "user",
                    "content": "Please analyze the sentiment of my 50 reviews and start a batch job.",
                },
                {
                    "role": "assistant",
                    "content": "I will inspect the available tools and verify the required input first.",
                    "tool_calls": [{"name": "list_tools", "arguments": {}}],
                },
                {
                    "role": "tool",
                    "name": "list_tools",
                    "content": "Available tools: create_batch_task, get_task_status. No sentiment-analysis tool is available.",
                },
                {
                    "role": "assistant",
                    "content": "The request refers to 50 reviews, but no review array was provided. I asked the user to provide the reviews instead of creating an empty task.",
                },
            ],
            "reference_task_id": reference.get("task_id"),
        },
        ensure_ascii=False,
    )


def parse_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8208/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen3 <think> output in the chat template.",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 <think> output and request direct JSON.",
    )
    parser.set_defaults(enable_thinking=False)
    args = parser.parse_args()

    reference = load_reference(args.skills)
    trajectory = build_trajectory(reference)
    prompt = (
        "Analyze the agent trajectory below and extract a reusable hindsight skill. "
        "Return JSON only with exactly these fields: episode_summary (string), "
        "episode_skill (string), and step_skills (array of objects with step_index and skill).\n\n"
        f"Trajectory:\n{trajectory}"
    )

    # Local vLLM endpoints must bypass any cluster-wide HTTP proxy.
    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        http_client=httpx.Client(trust_env=False),
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
    )
    text = response.choices[0].message.content or ""
    parsed = parse_json(text)
    required = {"episode_summary", "episode_skill", "step_skills"}
    result = {
        "model": args.model,
        "reference_task_id": reference.get("task_id"),
        "raw_output": text,
        "parsed": parsed,
        "schema_ok": bool(parsed and required.issubset(parsed)),
        "step_skills_ok": bool(parsed and isinstance(parsed.get("step_skills"), list)),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if not result["schema_ok"] or not result["step_skills_ok"]:
        raise SystemExit("model output did not match the candidate skill schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
