#!/usr/bin/env python3
"""Extract episode skills from AWM tool-use rollouts using vLLM endpoints."""
import argparse, json, os, re, threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import requests

from idea.analysis import SEEDEpisodeAnalyzer


def _tool_response_text(response):
    if response is None:
        return ""
    if isinstance(response, dict):
        return str(response.get("content", response))
    return str(response)


def _infer_tool_valid(message):
    """Infer execution validity from the tool result paired with this turn.

    AWM does not emit an action-valid flag. Tool errors are intentionally
    conservative: only explicit error-shaped responses are marked invalid;
    empty/missing responses remain unknown rather than being silently treated
    as valid.
    """
    if not message.get("tool_calls"):
        return None
    response = message.get("tool_response")
    if response is None:
        return None
    text = _tool_response_text(response).strip()
    if not text:
        return None
    # MCPToolExecutor prefixes result.isError responses with exactly
    # ``Error:`` (including validation, unknown-tool, and server errors).
    return not text.lower().startswith("error:")


def steps_from_rollout(r):
    # AWM traces are chat/tool transcripts rather than ALFWorld ``steps``.
    steps = []
    for i, msg in enumerate(r.get("trajectory") or r.get("messages") or []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        tool_valid = msg.get("tool_valid")
        if tool_valid is None:
            tool_valid = msg.get("info", {}).get("tool_valid") if isinstance(msg.get("info"), dict) else None
        if tool_valid is None:
            tool_valid = _infer_tool_valid(msg)
        tool_response = _tool_response_text(msg.get("tool_response"))
        calls = msg.get("tool_calls") or []
        response = str(content or "")
        if calls:
            response += "\nTool calls: " + json.dumps(calls, ensure_ascii=False)
        step = {"step_index": i, "observation": tool_response,
                "observation_prompt": tool_response, "response": response,
                "tool_valid": tool_valid,
                "task_description": r.get("task_description") or r.get("task", "")}
        # The shared AlfWorld formatter treats a present None as false via
        # bool(None); omit the compatibility field when validity is unknown.
        if tool_valid is not None:
            step["action_valid"] = tool_valid
        steps.append(step)
    return steps


def make_prompt(analyzer, r, steps):
    prompt = analyzer._build_episode_analysis_prompt(
        steps=steps, candidate_step_indices=[s["step_index"] for s in steps],
        analysis_mode="teacher_bootstrap", episode_success=1.0 if r.get("success") else 0.0,
        task_description=r.get("task_description") or r.get("task", ""))
    # Make the tool-use validity semantics explicit without changing analyzer schema.
    for m in prompt.get("messages", []):
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace("Action valid:", "Tool valid:")
    return prompt


def extract_one(r, endpoint, model, timeout, attempts):
    analyzer = SEEDEpisodeAnalyzer(backend="openai", max_completion_tokens=1024,
                                   max_step_skills_per_traj=0, skill_mode="episode_only")
    steps = steps_from_rollout(r)
    prompt = make_prompt(analyzer, r, steps)
    messages = [{"role": str(x.get("role", "user")), "content": str(x.get("content", ""))}
                for x in prompt.get("messages", [])]
    raw, err, parsed = "", None, {"episode_summary": "", "episode_skill": ""}
    for n in range(max(1, attempts)):
        try:
            resp = requests.post(endpoint.rstrip("/") + "/chat/completions",
                                 json={"model": model, "messages": messages, "temperature": 0.0,
                                       "max_tokens": 1024}, timeout=timeout)
            resp.raise_for_status(); raw = resp.json()["choices"][0]["message"]["content"]
            parsed = analyzer._parse_analysis_response(raw)
            if str(parsed.get("episode_skill", "")).strip():
                err = None; break
            err = "missing episode_skill"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return {"skill_id": f"{r.get('task_id','')}:{r.get('rollout_idx', r.get('rollout_id', 0))}",
            "task_id": r.get("task_id", ""), "task_type": r.get("scenario", "awm"),
            "source_rollout_id": r.get("rollout_idx", r.get("rollout_id", 0)),
            "source_success": bool(r.get("success", False)), "source_num_steps": len(steps),
            "task_description": r.get("task_description") or r.get("task", ""),
            "analysis_prompt": prompt, "llm_raw_output": raw,
            "episode_summary": parsed.get("episode_summary", ""),
            "episode_skill": parsed.get("episode_skill", ""), "parse_ok": bool(str(parsed.get("episode_skill", "")).strip()),
            "analysis_error": err}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--endpoints", default="http://127.0.0.1:2398/v1,http://127.0.0.1:2399/v1")
    ap.add_argument("--model", default="/local/user/Qwen3.8-27B"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=180); ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for line in out.open():
            try: done.add(json.loads(line)["skill_id"])
            except Exception: pass
    eps = [x.strip() for x in args.endpoints.split(",") if x.strip()]
    lock = threading.Lock(); counter = [0]
    def run(r):
        with lock: ep = eps[counter[0] % len(eps)]; counter[0] += 1
        return extract_one(r, ep, args.model, args.timeout, args.attempts)
    # Keep a bounded pool of in-flight requests. Refill immediately whenever
    # one request completes, so a long trajectory cannot stall the whole batch.
    processed = 0
    reported = 0
    progress_lock = threading.Lock()
    with out.open("a", buffering=1) as fo:
        with open(args.rollouts) as source:
            def pending_records():
                for line in source:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    sid = f"{r.get('task_id','')}:{r.get('rollout_idx', r.get('rollout_id', 0))}"
                    if sid not in done:
                        yield r

            records = pending_records()
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {}
                for _ in range(max(1, args.workers)):
                    try:
                        item = next(records)
                    except StopIteration:
                        break
                    futures[pool.submit(run, item)] = item
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in completed:
                        del futures[fut]
                        fo.write(json.dumps(fut.result(), ensure_ascii=False) + "\n"); fo.flush()
                        with progress_lock:
                            processed += 1
                            if processed // 10 > reported // 10:
                                reported = processed
                                print(f"progress={processed} output={out}", flush=True)
                        try:
                            item = next(records)
                        except StopIteration:
                            continue
                        futures[pool.submit(run, item)] = item
    print(f"processed={processed} resumed={len(done)} output={out}")

if __name__ == "__main__": main()
