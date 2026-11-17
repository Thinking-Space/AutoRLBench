#!/usr/bin/env python3
"""Convert ReTool SFT parquet data to Miles JSONL format."""

import argparse
import copy
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.retool_v2.tool_sandbox import tool_specs


def clean_messages(messages: list[dict]) -> list[dict]:
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        for call in tool_calls:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                arguments = {}
            function["arguments"] = {"code": arguments.get("code", "")}
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/user/panyinxu/datasets/ReTool-SFT-multi-turn/train_2000.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/user/panyinxu/datasets/miles_data/retool-sft-multi-turn-train-2000.jsonl"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pq.ParquetFile(args.input).read()
    rows = table.to_pylist()

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {
                "messages": clean_messages(row["messages"]),
                "tools": tool_specs,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
