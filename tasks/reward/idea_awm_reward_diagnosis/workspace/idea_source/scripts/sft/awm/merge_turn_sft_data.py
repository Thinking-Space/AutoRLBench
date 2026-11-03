#!/usr/bin/env python3
"""Clean existing AWM skill data and mix in turn-level tool-use examples."""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def remove_think(value: str) -> str:
    value = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.IGNORECASE | re.DOTALL)
    # Older Qwen traces sometimes contain only `</think>`; in trajectory
    # prompts, remove the hidden prefix of each serialized Response block.
    value = re.sub(r"(Response:\s*).*?</think>\s*", r"\1", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"</think>|<think>", "", value, flags=re.IGNORECASE)
    return value.strip()


def preserve_think(value: object) -> str:
    text = str(value or "").strip()
    close = text.lower().find("</think>")
    if close < 0:
        return text
    prefix = text[:close].strip()
    suffix = text[close + len("</think>"):].lstrip()
    if prefix.lower().startswith("<think>"):
        return text
    return f"<think>{prefix}</think>{suffix}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skill-data", required=True)
    p.add_argument("--turn-rollouts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    source = Path(args.skill_data)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    rows = []
    for item in frame.to_dict("records"):
        if str(item.get("task_type", "")) == "multi_turn_tool_use":
            continue
        rows.append({
            "prompt": remove_think(str(item["prompt"])),
            "response": remove_think(str(item["response"])),
            "task_id": str(item.get("task_id", "")),
            "task_type": str(item.get("task_type", "awm")),
        })

    turn_rows = 0
    with Path(args.turn_rollouts).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages") or []
            if not messages or messages[-1].get("role") != "assistant":
                continue
            history = []
            for message in messages[:-1]:
                role = str(message.get("role", "user"))
                history.append(f"{role}: {preserve_think(message.get('content', ''))}")
            response = preserve_think(messages[-1].get("content", ""))
            if not response:
                continue
            rows.append({
                "prompt": "You are completing a multi-turn tool-use task. Continue the conversation and produce the next assistant action or final answer.\n\n" + "\n\n".join(history),
                "response": response,
                "messages": [
                    {**m, "role": str(m.get("role", "user")),
                     "content": preserve_think(m.get("content", ""))}
                    for m in messages
                ],
                "task_id": f"turn_{turn_rows}",
                "task_type": "multi_turn_tool_use",
            })
            turn_rows += 1

    random.Random(args.seed).shuffle(rows)
    val_size = max(1, min(len(rows) - 1, int(round(len(rows) * args.val_ratio))))
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    pd.DataFrame(rows).to_parquet(output / "sft_episode_skill_all.parquet", index=False)
    pd.DataFrame(train_rows).to_parquet(output / "sft_episode_skill_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(output / "sft_episode_skill_val.parquet", index=False)
    (output / "build_summary.json").write_text(json.dumps({
        "total": len(rows), "train": len(train_rows), "val": len(val_rows),
        "skill_rows": len(rows) - turn_rows, "turn_rows": turn_rows,
    }, indent=2), encoding="utf-8")
    print(f"skill_rows={len(rows) - turn_rows} turn_rows={turn_rows} total={len(rows)} train={len(train_rows)} val={len(val_rows)}")


if __name__ == "__main__":
    main()
