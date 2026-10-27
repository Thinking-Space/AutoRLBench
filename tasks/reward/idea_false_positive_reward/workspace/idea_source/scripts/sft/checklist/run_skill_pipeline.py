#!/usr/bin/env python3
"""Extract and validate SEED-style skills from Checklist trajectories.

This is an offline pipeline for the Checklist cold-start corpus.  It resumes
from JSONL outputs and calls an OpenAI-compatible vLLM server.  Validation is
an LLM quality check against the recorded trajectory, not an environment
rollout success-rate measurement: Checklist has no executable ground-truth API.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


EXTRACTION_SYSTEM = """You are an expert agent-training analyst. Given a successful
tool-use trajectory, derive concise, reusable policy-facing skills. Preserve
conditional branches, tool-choice constraints, parameter validation, and error
recovery when they matter. Do not mention this specific conversation, UUIDs, or
invented tools. Return only JSON matching the requested schema."""

VALIDATION_SYSTEM = """You are a strict quality evaluator for agent-training
skills. Assess whether a proposed skill is supported by a recorded tool-use
trajectory and would improve an agent's future decisions. It must be specific,
actionable, faithful to the available tools and outcome, and must not leak
unobserved facts. Return only JSON matching the requested schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SEED-style baseline_rollouts.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2399/v1")
    parser.add_argument("--base-urls", default=None, help="Comma-separated endpoints; requests round-robin across them")
    parser.add_argument("--model", default="/local/user/Qwen3.8-27B")
    parser.add_argument("--stage", choices=("extract", "validate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-input-chars", type=int, default=45000)
    parser.add_argument("--max-completion-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def extract_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def trajectory_text(record: Dict[str, Any], max_chars: int) -> str:
    parts = [
        f"Task: {record.get('task_description', '')}",
        "Available tools:",
        json.dumps(record.get("tools", []), ensure_ascii=False),
        "Recorded interaction:",
    ]
    for index, message in enumerate(record.get("messages", [])):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        calls = message.get("tool_calls") or []
        suffix = f"\ntool_calls={json.dumps(calls, ensure_ascii=False)}" if calls else ""
        parts.append(f"[{index}] {role}: {content}{suffix}")
    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text
    # Retain task/tool context, beginning of the trajectory, and terminal outcome.
    head = text[: max_chars * 2 // 3]
    tail = text[-max_chars // 3 :]
    return head + "\n\n[trajectory truncated]\n\n" + tail


def request_json(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(min(5, max(1, retries))):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
            return extract_json(response.choices[0].message.content or "")
        except Exception as exc:  # request failures are persisted by the caller
            last_error = exc
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


SKILL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "episode_summary": {"type": "string"},
        "episode_skill": {"type": "string"},
        "step_skills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"step_index": {"type": "integer"}, "skill": {"type": "string"}},
                "required": ["step_index", "skill"],
            },
        },
    },
    "required": ["episode_summary", "episode_skill", "step_skills"],
}

VALIDATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "supported": {"type": "boolean"},
        "actionable": {"type": "boolean"},
        "generalizable": {"type": "boolean"},
        "faithful_to_tools": {"type": "boolean"},
        "quality_score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "actionable", "generalizable", "faithful_to_tools", "quality_score", "reason"],
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidate_skills.jsonl"
    validations_path = output_dir / "skill_validations.jsonl"
    source = read_jsonl(Path(args.input))
    if args.max_records is not None:
        source = source[: max(0, args.max_records)]
    source_by_id = {str(item["task_id"]): item for item in source}
    base_urls = [u.strip() for u in (args.base_urls or args.base_url).split(",") if u.strip()]
    clients = [OpenAI(base_url=u, api_key="EMPTY", timeout=args.timeout) for u in base_urls]
    write_lock = threading.Lock()

    if args.stage in {"extract", "all"}:
        completed = {str(item["task_id"]) for item in read_jsonl(candidates_path)}
        pending = [item for item in source if str(item["task_id"]) not in completed]

        def extract_one(item: Dict[str, Any]) -> Dict[str, Any]:
            task_id = str(item["task_id"])
            client = clients[hash(task_id) % len(clients)]
            prompt = (
                "Return a concise general episode skill and at most three critical step skills. "
                "Step skills must use assistant-step indices from the recorded interaction.\n\n"
                + trajectory_text(item, args.max_input_chars)
            )
            try:
                result = request_json(
                    client, model=args.model, system=EXTRACTION_SYSTEM, user=prompt,
                    schema_name="seed_skill", schema=SKILL_SCHEMA,
                    temperature=args.temperature, max_tokens=args.max_completion_tokens, retries=args.retries,
                )
                return {"task_id": task_id, "source_uuid": item.get("source_uuid"), "parse_ok": True, **result}
            except Exception as exc:
                return {"task_id": task_id, "source_uuid": item.get("source_uuid"), "parse_ok": False, "error": f"{type(exc).__name__}: {exc}"}

        print(f"extract: {len(pending)} pending / {len(source)} total", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for count, future in enumerate(concurrent.futures.as_completed([pool.submit(extract_one, item) for item in pending]), 1):
                record = future.result()
                append_jsonl(candidates_path, record, write_lock)
                if count % 10 == 0 or count == len(pending):
                    print(f"extract: {count}/{len(pending)} task_id={record['task_id']} parse_ok={record['parse_ok']}", flush=True)

    if args.stage in {"validate", "all"}:
        candidates = [item for item in read_jsonl(candidates_path) if item.get("parse_ok")]
        completed = {str(item["task_id"]) for item in read_jsonl(validations_path)}
        pending = [item for item in candidates if str(item["task_id"]) not in completed and str(item["task_id"]) in source_by_id]

        def validate_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
            task_id = str(candidate["task_id"])
            client = clients[hash(task_id) % len(clients)]
            trajectory = source_by_id[task_id]
            prompt = (
                "Evaluate the proposed skill against this recorded trajectory.\n\n"
                "Proposed skill:\n" + json.dumps({
                    "episode_skill": candidate.get("episode_skill", ""),
                    "step_skills": candidate.get("step_skills", []),
                }, ensure_ascii=False) + "\n\n" + trajectory_text(trajectory, args.max_input_chars)
            )
            try:
                result = request_json(
                    client, model=args.model, system=VALIDATION_SYSTEM, user=prompt,
                    schema_name="skill_validation", schema=VALIDATION_SCHEMA,
                    temperature=0.0, max_tokens=max(1024, args.max_completion_tokens), retries=args.retries,
                )
                accepted = bool(
                    result["supported"] and result["actionable"] and result["generalizable"]
                    and result["faithful_to_tools"] and float(result["quality_score"]) >= args.accept_threshold
                )
                return {"task_id": task_id, "source_uuid": candidate.get("source_uuid"), "accepted": accepted, **result}
            except Exception as exc:
                return {"task_id": task_id, "source_uuid": candidate.get("source_uuid"), "accepted": False, "error": f"{type(exc).__name__}: {exc}"}

        print(f"validate: {len(pending)} pending / {len(candidates)} extracted", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for count, future in enumerate(concurrent.futures.as_completed([pool.submit(validate_one, item) for item in pending]), 1):
                record = future.result()
                append_jsonl(validations_path, record, write_lock)
                if count % 10 == 0 or count == len(pending):
                    print(f"validate: {count}/{len(pending)} task_id={record['task_id']} accepted={record['accepted']}", flush=True)


if __name__ == "__main__":
    main()
