"""Multi-turn tool-using agent rollout for the public repository workload.

This is the ``--custom-generate-function-path`` used by the Miles rollout.  It
follows ``miles.rollout.generate_hub.multi_turn`` (same payload, routing header,
transport and token bookkeeping helpers) but resolves tools against the
per-trajectory environment returned by ``runtime.environment``.  Every engine
call is appended to a per-process JSONL trace so the run leaves a turn-level
record of prompt size, prefix-cache reuse and latency.
"""
from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.types import Sample

from . import environment, repo_tools

logger = logging.getLogger(__name__)
_TRACE_HANDLE = None


def _trace(record: dict) -> None:
    global _TRACE_HANDLE
    root = os.environ.get("RUN_EVIDENCE_DIR")
    if not root:
        return
    if _TRACE_HANDLE is None:
        path = Path(root) / "turns" / f"turns.{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _TRACE_HANDLE = path.open("a", buffering=1, encoding="utf-8")
    _TRACE_HANDLE.write(json.dumps(record, sort_keys=True) + "\n")


def _abort(sample: Sample, reason: str) -> GenerateFnOutput:
    sample.status = Sample.Status.ABORTED
    sample.reward = 0.0
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["abort_reason"] = reason
    if sample.response_length:
        sample.loss_mask = [0] * sample.response_length
    return GenerateFnOutput(samples=sample)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    metadata = sample.metadata or {}
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    parser = create_tool_call_parser(repo_tools.TOOL_SPECS, args.generate_tool_call_parser)
    session = sample.routing_key or f"{metadata.get('task_id')}-{sample.index}"
    started = time.time()

    try:
        env = await environment.open_environment(sample)
    except Exception as exc:  # environment failures must not become negative examples
        logger.warning("environment setup failed for %s: %r", session, exc)
        _trace({"event": "env_error", "session": session, "error": type(exc).__name__, "t": time.time()})
        return _abort(sample, f"env_error:{type(exc).__name__}")

    submission = None
    model_turns = 0
    try:
        sample.tokens = compute_prompt_ids_from_sample(input.state, sample, tools=repo_tools.TOOL_SPECS)
        for turn in range(args.generate_max_turns):
            payload, halt = compute_request_payload(args, sample.tokens, input.sampling_params)
            if payload is None:
                sample.status = halt
                break
            t0 = time.time()
            output = await input.state.engine_transport(url, payload, headers=compute_routing_headers(args, sample))
            model_turns += 1
            meta = output.get("meta_info", {})
            _trace({"event": "generate", "session": session, "turn": turn, "prompt_tokens": meta.get("prompt_tokens"),
                    "cached_tokens": meta.get("cached_tokens"), "completion_tokens": meta.get("completion_tokens"),
                    "latency_s": round(time.time() - t0, 4), "t": t0})
            await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
            finish = meta.get("finish_reason", {}).get("type")
            if finish == "abort":
                # the engine aborted the request (weight update); the partial trajectory
                # is not a valid training example, so hand it back for recycling
                _trace({"event": "engine_abort", "session": session, "turn": turn, "t": time.time()})
                return _abort(sample, "engine_abort")  # the finally clause closes the environment
            if finish == "length":
                break
            _, calls = parser.parse_non_stream(output["text"])
            if not calls:
                break
            final = [c for c in calls if c.name == repo_tools.FINAL_TOOL]
            if final:
                submission = json.loads(final[0].parameters or "{}")

            async def run_one(name, params):
                return await env.call(name, params)

            t1 = time.time()
            messages = await execute_tool_calls(calls, run_one)
            _trace({"event": "tools", "session": session, "turn": turn, "calls": [c.name for c in calls],
                    "latency_s": round(time.time() - t1, 4), "t": t1})
            if submission is not None:
                break
            update_sample_with_tool_responses(sample, messages, tokenizer=tokenizer)
    except environment.EnvironmentError_ as exc:
        _trace({"event": "env_error", "session": session, "error": type(exc).__name__, "t": time.time()})
        return _abort(sample, f"env_error:{type(exc).__name__}")
    finally:
        await env.close()

    sample.reward = repo_tools.score(sample.label, submission, model_turns)
    sample.metadata = dict(metadata, submitted=submission is not None, model_turns=model_turns,
                           wall_s=round(time.time() - started, 3))
    _trace({"event": "done", "session": session, "reward": sample.reward, "response_length": sample.response_length,
            "status": str(sample.status), "t": time.time()})
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser):
    parser.add_argument("--generate-max-turns", type=int, default=8)
    parser.add_argument("--generate-tool-call-parser", type=str, default="qwen3_coder")


generate.add_arguments = _add_arguments
