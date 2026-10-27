#!/usr/bin/env python3
"""Build prompt/response parquet files from AWM trajectories and candidate skills."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skills", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--max-prompt-chars", type=int, default=30000)
    p.add_argument("--compact", action="store_true", help="Use task metadata only as prompt context.")
    p.add_argument(
        "--turn-rollouts",
        default="",
        help="Optional turn-level SFT JSONL; its assistant turns are mixed with skill rows.",
    )
    return p.parse_args()


def json_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def remove_think(value: str) -> str:
    """Remove private reasoning from both trajectory context and targets."""
    value = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.IGNORECASE | re.DOTALL)
    # A few legacy traces contain only a closing marker.
    if "</think>" in value:
        value = value.split("</think>", 1)[1]
    return value.strip()


def prepend_reasoning(raw_output: object, response_json: str) -> str:
    """Keep reasoning before JSON and repair outputs containing only </think>."""
    raw = str(raw_output or "")
    close = raw.lower().find("</think>")
    if close < 0:
        return response_json
    prefix = raw[:close].strip()
    if prefix.lower().startswith("<think>"):
        prefix = prefix[len("<think>"):].strip()
    if not prefix:
        return response_json
    return f"<think>{prefix}</think>\n{response_json}"


def preserve_think(value: object) -> str:
    """Keep marked reasoning, adding a missing opening marker when needed."""
    text = json_text(value).strip()
    close = text.lower().find("</think>")
    if close < 0:
        return text
    prefix = text[:close].strip()
    suffix = text[close + len("</think>"):].lstrip()
    if prefix.lower().startswith("<think>"):
        return text
    return f"<think>{prefix}</think>{suffix}"


def clip_text(text: str, limit: int = 18000) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + "\n[tool output truncated]\n" + text[-tail:]


def trajectory_context(rollout: dict) -> str:
    # Match SEED's native analyzer format (seed.analysis._format_episode_steps):
    # represent the trajectory as natural-language step records.  Do not feed
    # the raw chat/tool transcript back as a nested chat message.
    parts = [f"Task description: {rollout.get('task_description') or rollout.get('task', '')}"]
    steps = rollout.get("steps") or []
    if not steps:
        steps = rollout.get("trajectory") or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_index = step.get("step_idx", step.get("step_index", index))
        observation = step.get("observation_prompt") or step.get("observation", "")
        response = step.get("model_response") or step.get("response") or step.get("content", "")
        action_valid = step.get("action_valid")
        block = (
            f"Step {step_index}\n"
            f"Observation: {clip_text(json_text(observation))}\n"
            f"Response: {clip_text(remove_think(json_text(response)))}"
        )
        if action_valid is not None:
            block += f"\nAction valid: {str(bool(action_valid)).lower()}"
        parts.append(block)
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    skill_path = Path(args.skills)
    rollout_path = Path(args.rollouts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skills = {}
    with skill_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("parse_ok", True) and item.get("task_id"):
                skills[str(item["task_id"])] = item

    rows = []
    with rollout_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rollout = json.loads(line)
            task_id = str(rollout.get("task_id", ""))
            skill = skills.get(task_id)
            if not skill:
                continue
            response = {
                "episode_summary": str(skill.get("episode_summary", "")),
                "episode_skill": str(skill.get("episode_skill", "")),
                "step_skills": skill.get("step_skills", []),
            }
            if args.compact:
                context = (
                    f"Scenario: {rollout.get('scenario', 'awm')}\n"
                    f"Task: {rollout.get('task_description') or rollout.get('task', '')}"
                )
            else:
                context = clip_text(trajectory_context(rollout), args.max_prompt_chars)
            prompt = (
                "Analyze the task below and write a concise, reusable episode-level "
                "policy skill. Include critical step-specific skills when present. "
                "Return JSON with episode_summary, episode_skill, and step_skills.\n\n"
                + context
            )
            response_json = json.dumps(response, ensure_ascii=False)
            rows.append({
                "prompt": prompt,
                "response": prepend_reasoning(skill.get("llm_raw_output", ""), response_json),
                "task_id": task_id,
                "task_type": rollout.get("scenario", "awm"),
            })

    turn_count = 0
    if args.turn_rollouts:
        # Turn-level records already contain system/user/tool history.  The
        # final assistant turn is the only supervised response for each row.
        with Path(args.turn_rollouts).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages") or []
                if not messages or messages[-1].get("role") != "assistant":
                    continue
                history = messages[:-1]
                prompt_parts = []
                for message in history:
                    role = str(message.get("role", "user"))
                    content = preserve_think(message.get("content", ""))
                    prompt_parts.append(f"{role}: {content}")
                response = preserve_think(messages[-1].get("content", ""))
                if not response:
                    continue
                rows.append({
                    "prompt": "You are completing a multi-turn tool-use task. Continue the conversation and produce the next assistant action or final answer.\n\n" + "\n\n".join(prompt_parts),
                    "response": response,
                    # Preserve the original structured conversation.  The
                    # mixed SFT dataset uses this field for turn-level rows;
                    # skill rows intentionally remain prompt/response text.
                    "messages": [
                        {**m, "role": str(m.get("role", "user")),
                         "content": preserve_think(m.get("content", ""))}
                        for m in history + [messages[-1]]
                    ],
                    "task_id": f"turn_{turn_count}",
                    "task_type": "multi_turn_tool_use",
                })
                turn_count += 1

    if args.max_records and args.max_records > 0:
        rows = rows[: args.max_records]
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 aligned records, found {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = max(1, min(len(rows) - 1, int(round(len(rows) * args.val_ratio))))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    pd.DataFrame(rows).to_parquet(output_dir / "sft_episode_skill_all.parquet", index=False)
    pd.DataFrame(train_rows).to_parquet(output_dir / "sft_episode_skill_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(output_dir / "sft_episode_skill_val.parquet", index=False)
    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"total": len(rows), "train": len(train_rows), "val": len(val_rows),
                   "skills": len(skills), "skill_rows": len(rows) - turn_count,
                   "turn_rows": turn_count,
                   "rollouts": sum(1 for _ in rollout_path.open())}, f, indent=2)
    print(f"aligned={len(rows)} train={len(train_rows)} val={len(val_rows)}")


if __name__ == "__main__":
    main()
