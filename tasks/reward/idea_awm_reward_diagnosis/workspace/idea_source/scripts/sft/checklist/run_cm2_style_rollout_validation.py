#!/usr/bin/env python3
"""Validate Checklist trajectories with CM2-style scripted-user tool rollouts.

This is an evaluation harness, not an RL trainer. It replays the user turns
and checklist annotations from the RL split while an OpenAI-compatible model
generates fresh assistant turns. Tool responses follow CM2's policy: replay an
exact recorded argument match when available, otherwise synthesize a JSON tool
result from the recorded examples. Checklist reward is evaluated per assistant
step, weighted, and only paid once per checklist item in each user turn.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


AGENT_SYSTEM = """You are a capable tool-using assistant. Follow the user's
request, call available tools whenever needed, and provide a concise final
answer after using tools. Do not invent tool outputs."""

TOOL_EXECUTOR_SYSTEM = """You are a precise tool executor. Given a function
schema, recorded examples, and a new call, return only the tool result as one
valid JSON object or array. Preserve the observed output format and types. Do
not explain your reasoning or mention examples."""

CHECKLIST_SYSTEM = """You are a precise checklist evaluator. Determine whether
the current user turn satisfies the supplied checklist item. Return only JSON
matching the requested schema. Be strict: require the stated focus and all
parts of the pass condition, and reject listed failure patterns."""

CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "boolean"}},
    "required": ["answer"],
}

JSON_VALUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"execution_result": {}},
    "required": ["execution_result"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw Checklist RL JSON with checklists")
    parser.add_argument("--output", required=True, help="Output rollout-validation JSONL")
    parser.add_argument("--base-url", default="http://127.0.0.1:2388/v1", help="Agent model endpoint (8B by default)")
    parser.add_argument("--model", default="/local/user/Qwen/Qwen3-8B", help="Agent model (8B by default)")
    parser.add_argument("--tool-base-url", default="http://127.0.0.1:2399/v1", help="Tool-result model endpoint")
    parser.add_argument("--tool-model", default="/local/user/Qwen3.8-27B")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:2399/v1", help="Checklist judge endpoint")
    parser.add_argument("--judge-model", default="/local/user/Qwen3.8-27B")
    parser.add_argument("--skills", help="Optional candidate_skills.jsonl, joined by UUID/task_id")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-assistant-steps-per-turn", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=18000)
    parser.add_argument("--tool-executor-tokens", type=int, default=1024)
    parser.add_argument("--judge-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--enable-thinking", action="store_true", help="Keep Qwen3 reasoning enabled for agent calls")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def parse_json(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise
        result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def request_json(
    client: OpenAI, model: str, system: str, user: str, schema_name: str,
    schema: Dict[str, Any], max_tokens: int, retries: int,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(min(5, max(1, retries))):
        raw_output: Optional[str] = None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max(1024, max_tokens),
                response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            )
            raw_output = response.choices[0].message.content or ""
            parsed = parse_json(raw_output)
            if trace is not None:
                trace.append({"attempt": attempt + 1, "ok": True, "raw_output": raw_output, "parsed": parsed})
            return parsed
        except Exception as exc:
            last_error = exc
            if trace is not None:
                trace.append({
                    "attempt": attempt + 1,
                    "ok": False,
                    "raw_output": raw_output,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def request_agent(client: OpenAI, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], args: argparse.Namespace) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(min(5, max(1, args.retries))):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=args.temperature,
                max_tokens=max(18000, args.max_completion_tokens),
                extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
            )
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def tool_index(messages: List[Dict[str, Any]]) -> Dict[str, List[Tuple[Dict[str, Any], str]]]:
    result: Dict[str, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        for offset, call in enumerate(calls):
            function = call.get("function", {}) if isinstance(call, dict) else {}
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (TypeError, json.JSONDecodeError):
                continue
            response_index = index + offset + 1
            if response_index >= len(messages) or messages[response_index].get("role") not in {"tool", "observation"}:
                continue
            result[str(function.get("name", ""))].append((arguments, str(messages[response_index].get("content", "{}"))))
    return result


def serialize_messages(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    turn = -1
    step = 0
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content") or ""
        if role == "user":
            turn += 1
            step = 0
            lines.append(f"# Turn: {turn}\nRole: user\ncontent: {content}\n")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            thinking_match = re.search(r"<think>(.*?)</think>", str(content), re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else "None"
            visible = str(content).split("</think>", 1)[-1].strip() if thinking_match else str(content).strip()
            lines.append(
                f"## Step: {step}\nRole: assistant\ncontent.thinking: {thinking}\n"
                f"content.user_visible_reply: {visible or 'None'}\ntool_call: {json.dumps(calls, ensure_ascii=False)}\n"
            )
            step += 1
        elif role in {"tool", "observation"}:
            lines.append(f"Role: tool\ncontent: {content}\n")
    return "".join(lines)


def checklist_prompt(previous_turns: List[Dict[str, Any]], current_turn: List[Dict[str, Any]], item: Dict[str, Any]) -> str:
    snippets = [e.get("snippet", "") for e in item.get("evidence", []) if isinstance(e, dict)]
    return (
        "# Criteria\n"
        f"Question: {item.get('question', '')}\nFocus on: {item.get('focus_on', '')}\n"
        f"Pass condition: {item.get('pass_condition', '')}\n"
        f"Failure examples: {json.dumps(item.get('failure_examples', []), ensure_ascii=False)}\n"
        f"Reference snippet: {json.dumps(snippets, ensure_ascii=False)}\n\n"
        "# Previous Messages\n" + serialize_messages(previous_turns) +
        "# Current Messages to Evaluate\n" + serialize_messages(current_turn)
    )


def normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    return [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict)]


def make_skill_system(skill: Optional[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    if not skill:
        return AGENT_SYSTEM
    episode_skill = str(skill.get("episode_skill", "")).strip()
    steps = skill.get("step_skills", [])
    text = episode_skill
    if isinstance(steps, list):
        text += "\n" + "\n".join(f"- {x.get('skill', '')}" for x in steps if isinstance(x, dict))
    tool_text = json.dumps(tools, ensure_ascii=False)
    return (AGENT_SYSTEM + "\n\nAvailable tools (call with <tool_call>{\"name\": ..., \"arguments\": {...}})</tool_call>:\n"
            + tool_text + "\n\nUse the following hindsight skill when applicable:\n" + text)


def parse_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """Read native OpenAI calls or Qwen's textual tool-call format."""
    calls: List[Dict[str, Any]] = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append({"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}})
    if calls:
        return calls
    content = str(getattr(message, "content", "") or "")
    candidates = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL | re.IGNORECASE)
    if not candidates:
        candidates = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.DOTALL | re.IGNORECASE)
    for index, candidate in enumerate(candidates):
        try:
            payload = json.loads(candidate)
            function = payload.get("function", payload)
            name = function.get("name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if name:
                calls.append({"id": f"text_call_{index}", "type": "function", "function": {"name": name, "arguments": arguments}})
        except (TypeError, json.JSONDecodeError):
            continue
    return calls


def run_one(record: Dict[str, Any], skill: Optional[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    task_id = str(record.get("uuid", ""))
    original = [dict(message) for message in record.get("messages", []) if isinstance(message, dict)]
    checklists = record.get("checklists")
    if not isinstance(checklists, list) or not checklists:
        return {"task_id": task_id, "success": False, "error": "missing checklists"}
    checklist = checklists[0]
    if not isinstance(checklist, list):
        return {"task_id": task_id, "success": False, "error": "malformed checklist"}
    users = [message for message in original if message.get("role") == "user"]
    if not users:
        return {"task_id": task_id, "success": False, "error": "missing user message"}

    agent_client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    tool_client = OpenAI(base_url=args.tool_base_url, api_key="EMPTY", timeout=args.timeout)
    judge_client = OpenAI(base_url=args.judge_base_url, api_key="EMPTY", timeout=args.timeout)
    recorded_tools = tool_index(original)
    tools = normalize_tools(record.get("tools"))
    messages: List[Dict[str, Any]] = [{"role": "system", "content": make_skill_system(skill, tools)}, dict(users[0])]
    turn_results: List[Dict[str, Any]] = []
    termination_reason = "completed_scripted_users"

    for turn_index, user_message in enumerate(users):
        if turn_index > 0:
            messages.append(dict(user_message))
        if turn_index >= len(checklist) or not isinstance(checklist[turn_index], list):
            termination_reason = "missing_turn_checklist"
            break
        turn_rules = checklist[turn_index]
        current_start = len(messages) - 1
        paid = [False] * len(turn_rules)
        required_passed = [not bool(rule.get("required_for_next_turn", False)) for rule in turn_rules]
        step_rewards: List[Dict[str, Any]] = []

        for step_index in range(args.max_assistant_steps_per_turn):
            response = request_agent(agent_client, args.model, messages, tools, args)
            message = response.choices[0].message
            tool_calls = parse_tool_calls(message)
            assistant = {"role": "assistant", "content": message.content or "", "tool_calls": tool_calls}
            messages.append(assistant)

            tool_errors = False
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    parameters = json.loads(call["function"]["arguments"])
                    if not isinstance(parameters, dict):
                        raise ValueError("tool arguments must decode to an object")
                    replay = next((content for known, content in recorded_tools.get(name, []) if canonical(known) == canonical(parameters)), None)
                    if replay is not None:
                        tool_result, source = replay, "recorded_exact_match"
                    else:
                        schema = next((tool for tool in tools if tool["function"].get("name") == name), None)
                        examples = recorded_tools.get(name, [])[:4]
                        executor_prompt = (
                            "Tool schema:\n" + json.dumps(schema or {}, ensure_ascii=False) +
                            "\nRecorded call-result examples:\n" + json.dumps(examples, ensure_ascii=False) +
                            "\nNew tool name: " + name + "\nNew arguments: " + json.dumps(parameters, ensure_ascii=False)
                        )
                        generated = request_json(tool_client, args.tool_model, TOOL_EXECUTOR_SYSTEM, executor_prompt, "tool_execution", JSON_VALUE_SCHEMA, args.tool_executor_tokens, args.retries)
                        tool_result, source = json.dumps(generated["execution_result"], ensure_ascii=False), "llm_synthesized"
                except Exception as exc:
                    tool_errors = True
                    tool_result, source = json.dumps({"error_tool_call": {"message": f"{type(exc).__name__}: {exc}"}}), "error"
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": tool_result, "tool_source": source})

            current_turn = messages[current_start:]
            checks: List[Dict[str, Any]] = []
            score = 0.0
            for rule_index, rule in enumerate(turn_rules):
                focus = rule.get("focus_on", "")
                if tool_errors and focus in {"assistant.tool_calls", "tool.content"}:
                    verdict, judge_ok = False, True
                    verdict_result = {"answer": False, "skipped": "tool_error"}
                    judge_attempts = [{"attempt": 0, "ok": False, "raw_output": None, "error": "skipped_due_to_tool_error"}]
                else:
                    judge_attempts: List[Dict[str, Any]] = []
                    try:
                        verdict_result = request_json(judge_client, args.judge_model, CHECKLIST_SYSTEM, checklist_prompt(messages[:current_start], current_turn, rule), "checklist_verdict", CHECK_SCHEMA, args.judge_tokens, args.retries, trace=judge_attempts)
                        verdict, judge_ok = bool(verdict_result["answer"]), True
                    except Exception as exc:
                        verdict, judge_ok = False, False
                        verdict_result = {"error": f"{type(exc).__name__}: {exc}"}
                if verdict and not paid[rule_index]:
                    score += float(rule.get("weight", 0.0))
                    paid[rule_index] = True
                if verdict and bool(rule.get("required_for_next_turn", False)):
                    required_passed[rule_index] = True
                checks.append({"id": rule.get("id"), "passed": verdict, "judge_ok": judge_ok, "weight": rule.get("weight"), "verdict": verdict_result, "judge_attempts": judge_attempts})
            step_rewards.append({"step_index": step_index, "reward": round(score, 8), "checks": checks, "tool_error": tool_errors})
            if not tool_calls:
                break
        else:
            termination_reason = "assistant_step_limit"

        turn_results.append({
            "turn_index": turn_index,
            "reward": round(sum(item["reward"] for item in step_rewards), 8),
            "required_passed": required_passed,
            "steps": step_rewards,
        })
        if not all(required_passed):
            termination_reason = "required_checklist_not_satisfied"
            break
        if termination_reason == "assistant_step_limit":
            break

    final_reward = round(sum(turn["reward"] for turn in turn_results), 8)
    result = {
        "task_id": task_id,
        "source_uuid": record.get("uuid"),
        "success": termination_reason == "completed_scripted_users",
        "termination_reason": termination_reason,
        "final_reward": final_reward,
        "turn_rewards": turn_results,
        "messages": messages,
        "skill": skill,
    }
    # Each trajectory creates three sync HTTP clients. Close them explicitly;
    # otherwise completed requests accumulate CLOSE-WAIT sockets at high
    # rollout concurrency and stall the process despite healthy model servers.
    agent_client.close()
    tool_client.close()
    judge_client.close()
    return result


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("input must be a top-level JSON array")
    if args.max_records is not None:
        source = source[:max(0, args.max_records)]
    skills: Dict[str, Dict[str, Any]] = {}
    if args.skills:
        for item in read_jsonl(Path(args.skills)):
            if item.get("parse_ok"):
                skills[str(item.get("source_uuid") or item.get("task_id"))] = item
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {str(item.get("task_id")) for item in read_jsonl(output)}
    pending = [item for item in source if str(item.get("uuid")) not in completed]
    print(f"rollout_validate: {len(pending)} pending / {len(source)} total", flush=True)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, item, skills.get(str(item.get("uuid"))), args) for item in pending]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            append_jsonl(output, result, lock)
            print(f"rollout_validate: {count}/{len(pending)} task_id={result.get('task_id')} reward={result.get('final_reward')} status={result.get('termination_reason', result.get('error'))}", flush=True)


if __name__ == "__main__":
    main()
