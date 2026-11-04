#!/usr/bin/env python3
"""Convert AWM skill-extraction JSONL records into SEED SFT parquet files."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def record_to_sft(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    if record.get("parse_ok") is False:
        return None
    prompt_obj = record.get("analysis_prompt")
    messages = prompt_obj.get("messages", []) if isinstance(prompt_obj, dict) else []
    if not messages:
        return None
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    content = last.get("content")
    skill = str(record.get("episode_skill", "")).strip()
    summary = str(record.get("episode_summary", "")).strip()
    if not isinstance(content, str) or not content.strip() or not skill or not summary:
        return None
    return {
        "prompt": content,
        "response": json.dumps({"episode_summary": summary, "episode_skill": skill}, ensure_ascii=False),
        "skill_id": str(record.get("skill_id", index)),
        "task_id": str(record.get("task_id", "")),
        "task_type": str(record.get("task_type", "")),
        "source_success": bool(record.get("source_success", False)),
        "source_num_steps": int(record.get("source_num_steps", 0) or 0),
        "parse_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    rows, skipped = [], 0
    with args.input.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = record_to_sft(json.loads(line), index)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
    if not rows:
        raise RuntimeError("No valid SFT records found")
    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * args.val_ratio))
    if len(shuffled) > 1 and args.val_ratio > 0:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    val_rows, train_rows = shuffled[:val_size], shuffled[val_size:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("all", rows), ("train", train_rows), ("val", val_rows)):
        with (args.output_dir / f"sft_episode_skill_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    import pandas as pd
    for name, data in (("all", rows), ("train", train_rows), ("val", val_rows)):
        pd.DataFrame(data).to_parquet(args.output_dir / f"sft_episode_skill_{name}.parquet", index=False)
    print(json.dumps({"input": str(args.input), "records": len(rows), "skipped": skipped, "train": len(train_rows), "val": len(val_rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
