#!/usr/bin/env python3
"""Convert Checklist conversations into SEED-compatible trajectory JSONL.

The converter preserves the original conversation and tool schemas, while
also creating the ``steps`` representation consumed by SEED's episode-skill
SFT pipeline.  It does not execute tools; tool calls and tool responses are
represented as recorded observations/actions.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Checklist JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--success-mode",
        choices=("true", "false", "unknown"),
        default="unknown",
        help=(
            "Success label for offline trajectories. The source files do not "
            "contain environment rewards; unknown is the safe default."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optionally convert only the first N records",
    )
    return parser.parse_args()


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _tool_call_payload(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    result: List[Dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        result.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": str(function.get("name", "")),
                    "arguments": arguments,
                },
            }
        )
    return result


def _tool_names(tools: Sequence[Any]) -> List[str]:
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def _stable_task_id(record: Dict[str, Any], index: int) -> str:
    value = str(record.get("uuid") or "")
    if value:
        return value
    user = next(
        (str(m.get("content", "")) for m in record.get("messages", [])
         if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    digest = hashlib.sha1(user.encode("utf-8")).hexdigest()[:16]
    return f"checklist-{index:06d}-{digest}"


def _first_user(messages: Sequence[Dict[str, Any]]) -> str:
    return next(
        (str(m.get("content", "")) for m in messages
         if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )


def _make_steps(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group each assistant turn with the preceding context and next result."""
    steps: List[Dict[str, Any]] = []
    history: List[str] = []
    step_idx = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = _json_text(message.get("content", ""))
        if role == "user":
            history.append(f"user: {content}")
            continue
        if role != "assistant":
            # Tool/observation output becomes the next observation. If no
            # assistant action follows, it is still retained in the history.
            history.append(f"{role}: {content}")
            if steps:
                steps[-1]["next_observation"] = "\n".join(history)
            continue

        calls = _tool_call_payload(message)
        action = {
            "tool_calls": calls,
            "content": content,
        } if calls else content
        observation = "\n".join(history)
        step: Dict[str, Any] = {
            "step_idx": step_idx,
            "observation": observation,
            "observation_prompt": observation,
            "model_response": content,
            "raw_action_text": _json_text(action),
            "executed_action": _json_text(action),
            "action_valid": bool(calls) if calls else True,
            "reward": 0.0,
            "done": False,
            "info": {
                "recorded_role": "assistant",
                "tool_call_count": len(calls),
            },
            "next_observation": "",
        }
        steps.append(step)
        history.append(f"assistant: {content}")
        step_idx += 1

    if steps:
        steps[-1]["done"] = True
        if not steps[-1]["next_observation"]:
            steps[-1]["next_observation"] = "\n".join(history)
    return steps


def convert_record(record: Dict[str, Any], index: int, success_mode: str) -> Dict[str, Any]:
    messages = [m for m in record.get("messages", []) if isinstance(m, dict)]
    tools = record.get("tools", [])
    if not isinstance(tools, list):
        try:
            tools = json.loads(tools)
        except (TypeError, json.JSONDecodeError):
            tools = []
    success: Optional[bool]
    if success_mode == "unknown":
        success = None
    else:
        success = success_mode == "true"
    task_id = _stable_task_id(record, index)
    steps = _make_steps(messages)
    for step in steps:
        step["task_description"] = _first_user(messages)
    return {
        "task_id": task_id,
        "task_type": "checklist_tool_call",
        "game_file": task_id,
        "rollout_id": 0,
        "seed": index,
        "task_description": _first_user(messages),
        "initial_info": {
            "source": "Checklist",
            "uuid": record.get("uuid"),
            "tool_names": _tool_names(tools),
            "offline_success_assumption": success_mode,
            "success_label_source": (
                "filtered_positive_demonstration" if success_mode == "true"
                else "explicit_cli_override" if success_mode == "false"
                else "not_available_in_source"
            ),
        },
        "tools": tools,
        "messages": messages,
        "steps": steps,
        "success": success,
        "completed": bool(steps),
        "num_steps": len(steps),
        "final_reward": 1.0 if success is True else 0.0,
        "baseline_key": f"{task_id}:0",
        "source_uuid": record.get("uuid"),
    }


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("Input must be a top-level JSON list")
    records: Iterable[Dict[str, Any]] = (
        item for item in source if isinstance(item, dict)
    )
    if args.max_records is not None:
        records = list(records)[: max(0, args.max_records)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(convert_record(record, index, args.success_mode), ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} trajectories to {output}")


if __name__ == "__main__":
    main()
