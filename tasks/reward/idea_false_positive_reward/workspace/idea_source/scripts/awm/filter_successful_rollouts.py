#!/usr/bin/env python3
"""Keep the final successful AWM rollout for every task id."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def derive_tools(record: dict) -> list[dict]:
    names: set[str] = set()
    for message in record.get("messages", []):
        for call in message.get("tool_calls", []) or []:
            args = call.get("arguments", {}) if isinstance(call, dict) else {}
            if isinstance(args, dict) and args.get("tool_name"):
                names.add(str(args["tool_name"]))
        content = message.get("content", "")
        if isinstance(content, str) and content.startswith("Available MCP Tools"):
            names.update(re.findall(r"^\d+\. (mcp_tool_[A-Za-z0-9_]+)", content, re.MULTILINE))
    return [{"name": name} for name in sorted(names)]


def main() -> None:
    args = parse_args()
    latest: dict[str, dict] = {}
    for line in Path(args.input).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record.get("task_id", ""))
        if task_id:
            latest[task_id] = record

    successful = [record for record in latest.values() if record.get("status") == "complete"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in successful:
            if not record.get("tools"):
                record["tools"] = derive_tools(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"final_records={len(latest)} successful={len(successful)} output={output}")


if __name__ == "__main__":
    main()
