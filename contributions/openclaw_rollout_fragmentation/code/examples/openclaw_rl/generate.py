"""Per-sample black-box OpenClaw rollout for ClawGym tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import secrets
import shlex
import socket
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from slime.agent.adapters import OpenClawAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.sandbox import Sandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .e2b_actor import RayE2BSandboxPool
from .rewards import get_reward_module
from .tasks import WORKSPACE, ClawGymTask, load_task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenClawConfig:
    adapter_public_url: str | None
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    image: str
    image_registry_type: str
    template: str
    cpu: str
    memory: str
    gateway_port: int
    context_window: int
    max_tokens: int
    # Reward policy for a session that fills the 128k context window and gets
    # truncated. Default keeps the grader's score; set
    # OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW=1 to score them 0 instead.
    zero_reward_on_context_overflow: bool
    agent_timeout_sec: int
    reward_timeout_sec: int
    rollout_guard_sec: int
    # Tail of the OpenClaw agent log to echo on a clean (exit=0) run. 0 disables.
    # Diagnostic for episodes that stop early without hitting any timeout.
    agent_log_tail_chars: int
    # Capture process/gateway/agent state before every sandbox teardown. This is
    # optional because the remote reads and large log records are useful for
    # investigations but expensive for a production run.
    runtime_diagnostics_enabled: bool
    # Hard wall-clock limit for best-effort runtime diagnostics during sandbox
    # cleanup. Diagnostics must never hold an aborted rollout open indefinitely.
    runtime_diagnostics_timeout_sec: float
    # Seconds between adapter accept-queue samples. 0 disables.
    adapter_socket_health_sec: float
    # Full OpenClaw session JSONL is large; keep it out of rollout samples unless
    # explicitly requested for debugging.
    save_transcript: bool
    # Best-effort budget for remote gateway/sandbox cleanup after diagnostics.
    # Remote RPCs may outlive cancellation, so this is a shared deadline rather
    # than an independent timeout for each cleanup operation.
    remote_cleanup_timeout_sec: float
    setup_attempts: int
    setup_backoff_base_sec: float
    e2b_actor_count: int | None
    e2b_actor_max_concurrency: int
    enable_thinking: bool
    replay_debug: bool
    search_provider: str
    search_plugin_dir: str
    search_api_key: str
    reader_plugin_dir: str
    reader_endpoint: str
    reader_api_key: str
    reader_work_project_id: str
    reader_proxy_project: str
    reader_proxy_sources: str
    # Advisor (edge-cloud) mode. When enabled the trainable edge actor may call a
    # parameterless `cloud_assist` tool that forwards its transcript to a fixed
    # external stronger model; the advice returns as a tool result (loss_mask=0,
    # never trained). The edge model only learns *when/how* to call the advisor
    # and what to do with the returned guidance.
    advisor_enabled: bool
    advisor_plugin_dir: str
    advisor_style: str
    cloud_model: str
    cloud_base_url: str
    cloud_api_key: str
    advisor_penalty_per_call: float
    advisor_penalty_cap: float
    advisor_reward_mode: str
    advisor_force_cloud: bool
    advisor_cost_noncached_input_per_1k_rmb: float
    advisor_cost_cached_input_per_1k_rmb: float
    advisor_cost_output_per_1k_rmb: float

    @classmethod
    def from_env(cls) -> OpenClawConfig:
        agent_timeout = int(os.environ.get("OPENCLAW_AGENT_TIMEOUT_SEC", "1800"))
        reward_timeout = int(os.environ.get("OPENCLAW_REWARD_TIMEOUT_SEC", "180"))
        guard = int(os.environ.get("OPENCLAW_ROLLOUT_GUARD_SEC", "0") or 0)
        actor_count_raw = os.environ.get("OPENCLAW_E2B_ACTOR_COUNT", "").strip()
        actor_count = int(actor_count_raw) if actor_count_raw else None
        if actor_count is not None and actor_count < 1:
            raise RuntimeError("OPENCLAW_E2B_ACTOR_COUNT must be positive")
        config = cls(
            adapter_public_url=os.environ.get("ADAPTER_PUBLIC_URL"),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "8666")),
            image=os.environ.get("E2B_IMAGE", "service.example.invalid/code-interpreter/openclaw:v2"),
            image_registry_type=os.environ.get("E2B_IMAGE_REGISTRY_TYPE", "enterprise"),
            template=os.environ.get("E2B_TEMPLATE", "openclaw"),
            cpu=os.environ.get("E2B_CPU", "2"),
            memory=os.environ.get("E2B_MEMORY", "4Gi"),
            gateway_port=int(os.environ.get("OPENCLAW_GATEWAY_PORT", "18789")),
            context_window=int(os.environ.get("OPENCLAW_CONTEXT_WINDOW", "32768")),
            max_tokens=int(os.environ.get("OPENCLAW_MAX_TOKENS", "4096")),
            zero_reward_on_context_overflow=os.environ.get(
                "OPENCLAW_ZERO_REWARD_ON_CONTEXT_OVERFLOW", "0"
            ).strip().lower()
            in ("1", "true", "yes"),
            agent_timeout_sec=agent_timeout,
            reward_timeout_sec=reward_timeout,
            rollout_guard_sec=guard or (agent_timeout + reward_timeout + 300),
            agent_log_tail_chars=int(os.environ.get("OPENCLAW_AGENT_LOG_TAIL_CHARS", "0")),
            runtime_diagnostics_enabled=os.environ.get(
                "OPENCLAW_RUNTIME_DIAGNOSTICS_ENABLED", "1"
            ).lower()
            in {"1", "true", "yes"},
            runtime_diagnostics_timeout_sec=float(
                os.environ.get("OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC", "30")
            ),
            adapter_socket_health_sec=float(os.environ.get("OPENCLAW_ADAPTER_SOCKET_HEALTH_SEC", "15")),
            save_transcript=os.environ.get("OPENCLAW_SAVE_TRANSCRIPT", "0").lower() in {"1", "true", "yes"},
            remote_cleanup_timeout_sec=float(
                os.environ.get("OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC", "30")
            ),
            setup_attempts=int(os.environ.get("OPENCLAW_E2B_SETUP_ATTEMPTS", "3")),
            setup_backoff_base_sec=float(os.environ.get("OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC", "1")),
            e2b_actor_count=actor_count,
            e2b_actor_max_concurrency=int(os.environ.get("OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY", "8")),
            enable_thinking=os.environ.get("OPENCLAW_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"},
            replay_debug=os.environ.get("OPENCLAW_REPLAY_DEBUG", "0").lower() in {"1", "true", "yes"},
            search_provider=os.environ.get("SEARCH_PROVIDER", "none").strip().lower(),
            search_plugin_dir=os.environ.get(
                "SEARCH_PLUGIN_DIR",
                "/workspace"
                "/QwenClawBench-e2b/openclaw_config/extensions/serp-search",
            ),
            search_api_key=os.environ.get("SERP_API_KEY", "").strip(),
            reader_plugin_dir=os.environ.get(
                "UDA_READER_PLUGIN_DIR",
                str(Path(__file__).resolve().parent / "plugins" / "uda-reader"),
            ),
            reader_endpoint=os.environ.get("UDA_READER_ENDPOINT", "").strip(),
            reader_api_key=os.environ.get("UDA_READER_API_KEY", "").strip(),
            reader_work_project_id=os.environ.get("UDA_READER_WORK_PROJECT_ID", "aux_model").strip(),
            reader_proxy_project=os.environ.get("UDA_READER_PROXY_PROJECT", "uda-reader").strip(),
            reader_proxy_sources=os.environ.get("UDA_READER_PROXY_SOURCES", "51").strip(),
            advisor_enabled=os.environ.get("OPENCLAW_ADVISOR", "0").lower() in {"1", "true", "yes"},
            advisor_plugin_dir=os.environ.get(
                "OPENCLAW_ADVISOR_PLUGIN_DIR",
                str(Path(__file__).resolve().parent / "plugins" / "cloud-assist-advisor-v2"),
            ),
            advisor_style=os.environ.get("ADVISOR_STYLE", "ace").lower(),
            cloud_model=os.environ.get("CLOUD_MODEL", ""),
            cloud_base_url=os.environ.get("CLOUD_BASE_URL", ""),
            cloud_api_key=os.environ.get("CLOUD_API_KEY", ""),
            advisor_penalty_per_call=float(os.environ.get("OPENCLAW_ADVISOR_PENALTY_PER_CALL", "0.05")),
            advisor_penalty_cap=float(os.environ.get("OPENCLAW_ADVISOR_PENALTY_CAP", "0.3")),
            advisor_reward_mode=os.environ.get("OPENCLAW_ADVISOR_REWARD_MODE", "static_penalty"),
            advisor_force_cloud=os.environ.get("OPENCLAW_ADVISOR_FORCE_CLOUD", "0").lower() in {"1", "true", "yes"},
            advisor_cost_noncached_input_per_1k_rmb=float(
                os.environ.get("OPENCLAW_ADVISOR_COST_NONCACHED_INPUT_PER_1K_RMB", "0.0014")
            ),
            advisor_cost_cached_input_per_1k_rmb=float(
                os.environ.get("OPENCLAW_ADVISOR_COST_CACHED_INPUT_PER_1K_RMB", "0.00026")
            ),
            advisor_cost_output_per_1k_rmb=float(os.environ.get("OPENCLAW_ADVISOR_COST_OUTPUT_PER_1K_RMB", "0.0044")),
        )
        get_reward_module(config.advisor_reward_mode)
        if (
            config.advisor_reward_mode in {"counterfactual_necessity", "counterfactual_cost_budget"}
            and not config.advisor_enabled
        ):
            raise RuntimeError(f"{config.advisor_reward_mode} reward requires OPENCLAW_ADVISOR=1")
        if config.advisor_force_cloud and not config.advisor_enabled:
            raise RuntimeError("OPENCLAW_ADVISOR_FORCE_CLOUD requires OPENCLAW_ADVISOR=1")
        if config.search_provider not in {"none", "disabled", "off", "serp"}:
            raise RuntimeError(f"unsupported SEARCH_PROVIDER={config.search_provider!r}; expected serp or none")
        if config.search_provider == "serp":
            required_web = {
                "SERP_API_KEY": config.search_api_key,
                "UDA_READER_ENDPOINT": config.reader_endpoint,
                "UDA_READER_API_KEY": config.reader_api_key,
                "UDA_READER_WORK_PROJECT_ID": config.reader_work_project_id,
                "UDA_READER_PROXY_PROJECT": config.reader_proxy_project,
                "UDA_READER_PROXY_SOURCES": config.reader_proxy_sources,
            }
            missing_web = [name for name, value in required_web.items() if not value]
            if missing_web:
                raise RuntimeError(
                    "OpenClaw web tools require environment variables: " + ", ".join(missing_web)
                )
        pricing = {
            "OPENCLAW_ADVISOR_COST_NONCACHED_INPUT_PER_1K_RMB": config.advisor_cost_noncached_input_per_1k_rmb,
            "OPENCLAW_ADVISOR_COST_CACHED_INPUT_PER_1K_RMB": config.advisor_cost_cached_input_per_1k_rmb,
            "OPENCLAW_ADVISOR_COST_OUTPUT_PER_1K_RMB": config.advisor_cost_output_per_1k_rmb,
        }
        invalid_pricing = [name for name, value in pricing.items() if not math.isfinite(value) or value < 0]
        if invalid_pricing:
            raise RuntimeError("advisor cost pricing must be finite and non-negative: " + ", ".join(invalid_pricing))
        if config.advisor_enabled:
            required = {
                "CLOUD_MODEL": config.cloud_model,
                "CLOUD_BASE_URL": config.cloud_base_url,
                "CLOUD_API_KEY": config.cloud_api_key,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise RuntimeError(
                    "OPENCLAW_ADVISOR is enabled, but required environment variables are empty: " + ", ".join(missing)
                )
        if config.setup_attempts < 1:
            raise RuntimeError("OPENCLAW_E2B_SETUP_ATTEMPTS must be positive")
        if not math.isfinite(config.setup_backoff_base_sec) or config.setup_backoff_base_sec < 0:
            raise RuntimeError("OPENCLAW_E2B_SETUP_BACKOFF_BASE_SEC must be finite and non-negative")
        if not math.isfinite(config.runtime_diagnostics_timeout_sec) or config.runtime_diagnostics_timeout_sec <= 0:
            raise RuntimeError("OPENCLAW_RUNTIME_DIAGNOSTICS_TIMEOUT_SEC must be finite and positive")
        if not math.isfinite(config.remote_cleanup_timeout_sec) or config.remote_cleanup_timeout_sec <= 0:
            raise RuntimeError("OPENCLAW_REMOTE_CLEANUP_TIMEOUT_SEC must be finite and positive")
        if config.e2b_actor_max_concurrency < 1:
            raise RuntimeError("OPENCLAW_E2B_ACTOR_MAX_CONCURRENCY must be positive")
        return config


# The trainable edge actor sees this appended to its user query when advisor mode
# is on. Mirrors claw_ec_evaluation's lib_advisor ADVISOR_PROMPT with the tool
# name switched from `advisor` to the v2 plugin's `cloud_assist`.
ADVISOR_PROMPT = """

You have access to a `cloud_assist` tool backed by a stronger reviewer model. It takes NO parameters — when you call cloud_assist(), your entire conversation history is automatically forwarded. They see the task, every tool call you've made, every result you've seen.

Call cloud_assist BEFORE substantive work — before writing, before committing to an interpretation, before building on an assumption. If the task requires orientation first (finding files, fetching a source, seeing what's there), do that, then call cloud_assist. Orientation is not substantive work. Writing, editing, and declaring an answer are.

Also call cloud_assist:
- When you believe the task is complete. BEFORE this call, make your deliverable durable: write the file, save the result, commit the change. The cloud_assist call takes time; if the session ends during it, a durable result persists and an unwritten one doesn't.
- When stuck — errors recurring, approach not converging, results that don't fit.
- When considering a change of approach.

On tasks longer than a few steps, call cloud_assist at least once before committing to an approach and once before declaring done. On short reactive tasks where the next action is dictated by tool output you just read, you don't need to keep calling — the advisor adds most of its value on the first call, before the approach crystallizes.

Give the advice serious weight. If you follow a step and it fails empirically, or you have primary-source evidence that contradicts a specific claim (the file says X, the paper states Y), adapt. A passing self-test is not evidence the advice is wrong — it's evidence your test doesn't check what the advice is checking.

If you've already retrieved data pointing one way and the advisor points another: don't silently switch. Surface the conflict in one more cloud_assist call — "I found X, you suggest Y, which constraint breaks the tie?" The advisor saw your evidence but may have underweighted it; a reconcile call is cheaper than committing to the wrong branch.
"""

# Fixed sandbox paths shared with the cloud-assist-advisor plugin (see its index.js).
ADVISOR_CTX_PATH = "/dev/shm/.advisor_ctx.bin"
ADVISOR_AUDIT_PATH = "/tmp/cloud_assistant_audit.jsonl"
ADVISOR_TRACE_PATH = "/tmp/cloud_assistant_trace.jsonl"


CONFIG = OpenClawConfig.from_env()
REWARD_MODULE = get_reward_module(CONFIG.advisor_reward_mode)
_E2B_ACTOR_POOL: RayE2BSandboxPool | None = None


@dataclass(frozen=True)
class _ReplaySnapshot:
    token_ids: list[int]
    prompt_tokens: int
    output_tokens: int
    generated_reasoning: bool
    manager_message: dict[str, Any]


@dataclass(frozen=True)
class AgentRunResult:
    exit_code: int | None
    log: str
    truncated: bool


def _timeline_event(timeline: dict[str, Any] | None, event: str, **fields: Any) -> None:
    """Record rollout lifecycle events with both wall-clock and relative time."""
    if timeline is None:
        return
    now = time.time()
    item = {
        "event": event,
        "time_unix": now,
        "elapsed_sec": round(now - timeline["started_at"], 3),
        **fields,
    }
    timeline["events"].append(item)
    logger.info("[openclaw_rl] episode_timeline %s", json.dumps(item, sort_keys=True, default=str))


@dataclass(frozen=True)
class RewardRunResult:
    score: float
    verifier_failed: bool


class AgentProcessError(RuntimeError):
    def __init__(self, exit_code: int, log: str) -> None:
        super().__init__(f"OpenClaw agent exited {exit_code}: {log[-2000:]}")
        self.exit_code = exit_code
        self.log = log


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _replay_diagnostic(snapshot: _ReplaySnapshot, current_prompt_ids: list[int], messages: list[dict]) -> dict[str, Any]:
    common_prefix = _common_prefix_length(snapshot.token_ids, current_prompt_ids)
    latest_assistant = next(
        (message for message in reversed(messages) if isinstance(message, dict) and message.get("role") == "assistant"),
        None,
    )
    reasoning_field_present = isinstance(latest_assistant, dict) and "reasoning_content" in latest_assistant
    replayed_reasoning = bool(latest_assistant and latest_assistant.get("reasoning_content"))
    echoed_without_reasoning = (
        {key: value for key, value in latest_assistant.items() if key != "reasoning_content"}
        if isinstance(latest_assistant, dict)
        else None
    )
    manager_echo_exact = snapshot.manager_message == latest_assistant
    manager_echo_without_reasoning_exact = snapshot.manager_message == echoed_without_reasoning

    mismatch_fields: list[str] = []
    if isinstance(latest_assistant, dict):
        for key in sorted(set(snapshot.manager_message) | set(latest_assistant)):
            if snapshot.manager_message.get(key) != latest_assistant.get(key):
                mismatch_fields.append(key)
    else:
        mismatch_fields.append("missing_assistant")

    if common_prefix < snapshot.prompt_tokens:
        drift_region = "previous_prompt"
    elif common_prefix < len(snapshot.token_ids):
        drift_region = "previous_output"
    else:
        drift_region = "after_previous_output"

    return {
        "previous_prompt_tokens": snapshot.prompt_tokens,
        "previous_output_tokens": snapshot.output_tokens,
        "previous_generated_reasoning": snapshot.generated_reasoning,
        "current_prompt_tokens": len(current_prompt_ids),
        "common_prefix_tokens": common_prefix,
        "previous_output_tokens_matched": max(
            0,
            min(snapshot.output_tokens, common_prefix - snapshot.prompt_tokens),
        ),
        "exact_previous_prefix": common_prefix == len(snapshot.token_ids),
        "replayed_reasoning_field_present": reasoning_field_present,
        "replayed_reasoning_nonempty": replayed_reasoning,
        "manager_echo_exact": manager_echo_exact,
        "manager_echo_without_reasoning_exact": manager_echo_without_reasoning_exact,
        "manager_echo_mismatch_fields": mismatch_fields,
        "drift_region": drift_region,
    }


def _generated_output_has_reasoning(tokenizer, output_ids: list[int], reasoning_parser_name: str | None) -> bool:
    if not reasoning_parser_name or not output_ids:
        return False
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    try:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning, _ = ReasoningParser(
            model_type=reasoning_parser_name,
            stream_reasoning=False,
        ).parse_non_stream(raw_output)
        return bool(reasoning) or "<think>" in raw_output or "</think>" in raw_output
    except Exception:
        logger.exception("[openclaw-replay] failed to parse generated reasoning")
        return "<think>" in raw_output or "</think>" in raw_output


def _trajectory_fork_threshold_tokens() -> int:
    """Return the trajectory drift policy configured for this run."""
    value = int(os.environ.get("OPENCLAW_FORK_THRESHOLD_TOKENS", "1024"))
    if value < 0:
        raise ValueError("OPENCLAW_FORK_THRESHOLD_TOKENS must be non-negative")
    return value


def _adapter_accept_queue(port: int) -> tuple[int, int | None] | None:
    """Current accept-queue depth of the listening adapter socket, and its cap.

    ``ss`` reports both (Recv-Q / Send-Q) but is not guaranteed to be installed;
    ``/proc/net/tcp`` always carries the depth (``sk_ack_backlog`` lands in
    rx_queue for TCP_LISTEN) but not the cap, so fall back to depth alone.
    """
    try:
        listing = subprocess.run(
            ["ss", "-lntH", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        for row in listing.splitlines():
            fields = row.split()
            if len(fields) >= 3:
                return int(fields[1]), int(fields[2])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    try:
        entries = Path("/proc/net/tcp").read_text().splitlines()[1:]
        entries += Path("/proc/net/tcp6").read_text().splitlines()[1:]
    except OSError:
        return None
    for entry in entries:
        fields = entry.split()
        if len(fields) < 5 or fields[3] != "0A":  # 0A = TCP_LISTEN
            continue
        if int(fields[1].rsplit(":", 1)[1], 16) != port:
            continue
        return int(fields[4].split(":")[1], 16), None
    return None


def _probe_adapter_locally(port: int, timeout_sec: float = 5.0) -> tuple[bool, float]:
    """Round-trip /healthz over loopback, bypassing the gateway."""
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_sec) as sock:
            sock.sendall(b"GET /healthz HTTP/1.0\r\nHost: localhost\r\n\r\n")
            sock.settimeout(timeout_sec)
            served = b"200" in sock.recv(64)
    except OSError:
        return False, time.monotonic() - started
    return served, time.monotonic() - started


def _watch_adapter_socket(port: int, interval_sec: float) -> None:
    """Record why the gateway cannot reach the adapter.

    OpenClaw abandons a run on the first 5xx, so every gateway-side
    ``delayed connect error: 111`` freezes a trajectory on an unanswered tool
    call -- measured 2026-08-03 at ~70% of probe requests during rollout against
    0% on an idle single-pod job. ECONNREFUSED on a socket that is still
    listening means its accept queue overflowed, so the pair to watch is
    ``queue`` against ``backlog``: they meet exactly when connections start
    getting refused. ``local_ms`` separates the two candidate culprits -- a slow
    loopback round trip means the adapter's event loop stalled (its per-turn
    tokenisation is synchronous), while a fast one points at the gateway.
    """
    while True:
        time.sleep(interval_sec)
        try:
            queue = _adapter_accept_queue(port)
            served, elapsed = _probe_adapter_locally(port)
        except Exception:
            # A dead diagnostic thread is worse than a noisy one: keep sampling.
            logger.exception("[openclaw-adapter] socket_health sample failed")
            continue
        logger.info(
            "[openclaw-adapter] socket_health port=%d queue=%s backlog=%s local_ok=%s local_ms=%.0f",
            port,
            queue[0] if queue else "?",
            queue[1] if queue else "?",
            served,
            elapsed * 1000,
        )


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        if not CONFIG.adapter_public_url and not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_URL or ADAPTER_PUBLIC_HOST must point to an adapter endpoint reachable from E2B"
            )
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        reasoning_parser_name = getattr(args, "sglang_reasoning_parser", None) or None
        self.last_messages: dict[str, str] = {}
        self.advisor_call_output_ids: dict[str, list[list[int]]] = defaultdict(list)
        self.last_turn_tool_calls: dict[str, bool] = {}
        self.last_turn_output_len: dict[str, int] = {}
        self.replay_snapshots: dict[str, _ReplaySnapshot] = {}
        self.replay_turns: dict[str, int] = defaultdict(int)
        self.agent_turn_counts: dict[str, int] = defaultdict(int)
        self.agent_tool_call_counts: dict[str, int] = defaultdict(int)
        self.agent_model_request_counts: dict[str, int] = defaultdict(int)
        self.agent_model_response_counts: dict[str, int] = defaultdict(int)
        self.agent_generated_token_counts: dict[str, int] = defaultdict(int)
        self.agent_max_context_token_counts: dict[str, int] = defaultdict(int)
        self.advisor_followup_request_counts: dict[str, int] = defaultdict(int)
        self.advisor_followup_response_counts: dict[str, int] = defaultdict(int)

        def capture_request_start(sid, body, request_index) -> None:
            self.agent_model_request_counts[sid] += 1
            messages = body.get("messages", []) if isinstance(body, dict) else []
            has_cloud_assistant_call = any(
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and any(
                    isinstance(call, dict)
                    and isinstance(call.get("function"), dict)
                    and (
                        call["function"].get("name") == "cloud_assist"
                        or call.get("name") == "cloud_assist"
                    )
                    for call in message.get("tool_calls", [])
                )
                for message in messages
            )
            has_advisor_result = any(
                isinstance(message, dict)
                and message.get("role") == "tool"
                and (
                    message.get("name") == "cloud_assist"
                    or message.get("tool_name") == "cloud_assist"
                    or (has_cloud_assistant_call and message.get("tool_call_id"))
                )
                for message in messages
            )
            if has_advisor_result:
                self.advisor_followup_request_counts[sid] += 1
            logger.info(
                "[openclaw-adapter] sid=%s model_request_start index=%s has_cloud_tool_result=%s",
                sid,
                request_index,
                has_advisor_result,
            )

        def capture_last_message(sid, messages, _tools, manager_message, turn) -> None:
            if CONFIG.replay_debug:
                self.replay_turns[sid] += 1
                turn_index = self.replay_turns[sid]
                previous = self.replay_snapshots.get(sid)
                diagnostic: dict[str, Any] = {
                    "sid": sid,
                    "turn": turn_index,
                    "comparison_available": previous is not None,
                }
                if previous is not None:
                    diagnostic.update(_replay_diagnostic(previous, list(turn.prompt_ids), messages))

                generated_reasoning = _generated_output_has_reasoning(
                    tokenizer,
                    list(turn.output_ids),
                    reasoning_parser_name,
                )
                diagnostic.update(
                    {
                        "current_output_tokens": len(turn.output_ids),
                        "current_generated_reasoning": generated_reasoning,
                    }
                )
                logger.warning("[openclaw-replay] %s", json.dumps(diagnostic, sort_keys=True))
                self.replay_snapshots[sid] = _ReplaySnapshot(
                    token_ids=list(turn.prompt_ids) + list(turn.output_ids),
                    prompt_tokens=len(turn.prompt_ids),
                    output_tokens=len(turn.output_ids),
                    generated_reasoning=generated_reasoning,
                    manager_message=dict(manager_message),
                )

            self.agent_turn_counts[sid] += 1
            self.agent_model_response_counts[sid] += 1
            self.agent_generated_token_counts[sid] += len(turn.output_ids)
            self.agent_max_context_token_counts[sid] = max(
                self.agent_max_context_token_counts[sid],
                len(turn.prompt_ids) + len(turn.output_ids),
            )
            content = manager_message.get("content") if isinstance(manager_message, dict) else None
            if content:
                self.last_messages[sid] = str(content)
            tool_calls = manager_message.get("tool_calls", []) if isinstance(manager_message, dict) else []
            self.agent_tool_call_counts[sid] += sum(1 for call in tool_calls if isinstance(call, dict))
            # Structured view of the newest turn, for the dangling-turn mask:
            # the adapter already parsed the output, so whether it ended on tool
            # calls needs no re-decoding, and the output length lets the masker
            # verify it found the same turn before touching any loss mask.
            self.last_turn_tool_calls[sid] = bool(tool_calls)
            self.last_turn_output_len[sid] = len(turn.output_ids)
            if any(
                isinstance(call, dict)
                and isinstance(call.get("function"), dict)
                and call["function"].get("name") == "cloud_assist"
                for call in tool_calls
            ):
                self.advisor_call_output_ids[sid].append(list(turn.output_ids))

            if self.advisor_followup_request_counts[sid] > self.advisor_followup_response_counts[sid]:
                self.advisor_followup_response_counts[sid] += 1

        self.adapter = OpenClawAdapter(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=reasoning_parser_name,
            fork_threshold_tokens=_trajectory_fork_threshold_tokens(),
            apply_chat_template_kwargs={"enable_thinking": CONFIG.enable_thinking},
            debug_callback=capture_last_message,
            request_debug_callback=capture_request_start,
            require_registered_session=True,
            # Length is capped at rollout time (32k per request, 128k per
            # session) and the trajectory is truncated at the window. Keep the
            # grader's score for those runs instead of zeroing it: measured on
            # this task set, sessions that hit the cap scored *above* average
            # (mean 0.360 vs 0.346, 28.5% zero vs 32.3%), so zeroing them would
            # penalise length rather than failure.
            zero_reward_on_overflow=CONFIG.zero_reward_on_context_overflow,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="openclaw-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        logger.info(
            "[openclaw-adapter] listening host=%s port=%d node=%s forwarder_enabled=%s",
            CONFIG.adapter_bind_host,
            self.app_handle.port,
            socket.gethostname(),
            os.environ.get("ADAPTER_FORWARDER_ENABLED", "0"),
        )
        self.adapter_url = _openai_base_url(
            CONFIG.adapter_public_url
            if CONFIG.adapter_public_url
            else f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        )
        if CONFIG.adapter_socket_health_sec > 0:
            threading.Thread(
                target=_watch_adapter_socket,
                args=(self.app_handle.port, CONFIG.adapter_socket_health_sec),
                name="openclaw-adapter-socket-health",
                daemon=True,
            ).start()

    def pop_last_message(self, sid: str) -> str:
        return self.last_messages.pop(sid, "")

    def pop_last_turn_shape(self, sid: str) -> tuple[bool, int]:
        """(had_tool_calls, output_len) of the newest recorded turn."""
        return (
            self.last_turn_tool_calls.pop(sid, False),
            self.last_turn_output_len.pop(sid, 0),
        )

    def pop_advisor_call_output_ids(self, sid: str) -> list[list[int]]:
        return self.advisor_call_output_ids.pop(sid, [])

    def pop_agent_turn_stats(self, sid: str) -> dict[str, int]:
        return {
            "turns": self.agent_turn_counts.pop(sid, 0),
            "tool_calls": self.agent_tool_call_counts.pop(sid, 0),
            "model_requests": self.agent_model_request_counts.pop(sid, 0),
            "model_responses": self.agent_model_response_counts.pop(sid, 0),
            "generated_tokens": self.agent_generated_token_counts.pop(sid, 0),
            "max_context_tokens": self.agent_max_context_token_counts.pop(sid, 0),
            "advisor_followup_requests": self.advisor_followup_request_counts.pop(sid, 0),
            "advisor_followup_responses": self.advisor_followup_response_counts.pop(sid, 0),
        }

    def clear_replay_probe(self, sid: str) -> None:
        self.replay_snapshots.pop(sid, None)
        self.replay_turns.pop(sid, None)



def _mask_dangling_final_turn(
    samples: list[Sample],
    *,
    last_turn_tool_calls: bool,
    last_turn_output_len: int,
) -> int:
    """Zero the loss mask of a trailing tool-call turn that never got its result.

    A trajectory that ends on a tool call was cut by something external (agent
    deadline, context overflow, a dropped connection): OpenClaw always executes
    a tool call it received, so a natural completion ends in plain text. The
    consequence of that final call is unobservable -- its tool result never
    entered any prompt and reward.sh scored artifacts it did not influence --
    so training those tokens on the episode score attributes an unrelated
    number to them. Every earlier turn did get its feedback and keeps its mask.

    The verdict comes from the adapter's own parse of the newest turn
    (``last_turn_tool_calls``), not from re-decoding strings, so it holds for
    any output format. ``last_turn_output_len`` must match the final trained
    run exactly or nothing is touched: a mismatch means the run under the
    cursor is not the turn the adapter recorded (fragment-ordering surprise,
    overflow truncation mid-turn), and a wrong mask is worse than none.

    Returns the number of response tokens masked (0 when the trajectory ended
    naturally). Only the final fragment can carry the dangling turn; earlier
    fragments are forked prefixes whose turns all saw their results.
    """
    if not samples or not last_turn_tool_calls or last_turn_output_len <= 0:
        return 0
    sample = samples[-1]
    mask = sample.loss_mask
    response_length = int(sample.response_length or 0)
    if not mask or response_length <= 0:
        return 0
    end = len(mask)
    while end > 0 and not mask[end - 1]:
        end -= 1
    if end == 0:
        return 0
    start = end
    while start > 0 and mask[start - 1]:
        start -= 1
    if end != len(mask) or end - start != last_turn_output_len:
        # A trained run after the final turn's tokens, or a run of a different
        # length, means this is not the recorded final turn. Leave it alone.
        return 0
    for position in range(start, end):
        mask[position] = 0
    return end - start


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    del evaluation
    state = _AdapterService(args)
    reward_role = REWARD_MODULE.rollout_role(args, base_sample) if CONFIG.advisor_enabled else "policy"
    task_id = str(base_sample.label or "")
    try:
        task = load_task(base_sample)
        task_id = task.task_id
    except Exception as exc:
        return _abort_result(
            base_sample,
            f"invalid_task:{type(exc).__name__}",
            task_id=task_id,
            reward_role=reward_role,
        )

    session_id = base_sample.session_id or _session_id(base_sample, task.task_id)
    base_sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=CONFIG.context_window,
    )
    rollout_advisor_enabled = CONFIG.advisor_enabled and reward_role != "local_baseline"

    started = time.time()
    timeline: dict[str, Any] = {"started_at": started, "events": []}
    _timeline_event(timeline, "episode_start", task_id=task.task_id, session_id=session_id)
    agent_result: AgentRunResult | None = None
    reward_result: RewardRunResult | None = None
    advisor_stats: dict[str, Any] | None = None
    transcript_holder: dict[str, list[dict[str, Any]]] = {}
    agent_turn_stats: dict[str, int] = {
        "turns": 0,
        "tool_calls": 0,
        "model_requests": 0,
        "model_responses": 0,
        "generated_tokens": 0,
        "max_context_tokens": 0,
        "advisor_followup_requests": 0,
        "advisor_followup_responses": 0,
    }
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with _prepared_rollout_sandbox(
                args,
                task,
                session_id,
                state.adapter_url,
                advisor_enabled=rollout_advisor_enabled,
                timeline=timeline,
                transcript_holder=transcript_holder,
            ) as sb:
                try:
                    _timeline_event(timeline, "agent_wait_start")
                    agent_result = await _run_agent(sb, timeline=timeline)
                    _timeline_event(
                        timeline,
                        "agent_exit",
                        exit_code=agent_result.exit_code,
                        truncated=agent_result.truncated,
                    )

                    final_message = state.pop_last_message(session_id) or agent_result.log.strip()
                    _timeline_event(timeline, "reward_start")
                    reward_result = await _run_reward(sb, task, final_message)
                    _timeline_event(
                        timeline,
                        "reward_end",
                        score=reward_result.score,
                        verifier_failed=reward_result.verifier_failed,
                    )
                    advisor_stats = await _collect_advisor_usage(sb) if rollout_advisor_enabled else None
                    advisor_call_output_ids = state.pop_advisor_call_output_ids(session_id)
                    agent_turn_stats = state.pop_agent_turn_stats(session_id)
                    agent_turns = agent_turn_stats["turns"]
                    agent_tool_calls = agent_turn_stats["tool_calls"]
                    if CONFIG.advisor_force_cloud and rollout_advisor_enabled and not advisor_stats["calls"]:
                        raise RuntimeError("forced-cloud rollout did not call cloud_assist")
                    if CONFIG.advisor_force_cloud and rollout_advisor_enabled:
                        if advisor_stats["failed_calls"] or any(
                            cost <= 0 for cost in advisor_stats["call_costs_rmb"]
                        ):
                            raise RuntimeError("forced-cloud rollout has a failed call or missing positive cost data")
                    if (
                        CONFIG.advisor_reward_mode in {"counterfactual_necessity", "counterfactual_cost_budget"}
                        and advisor_stats is not None
                    ):
                        if advisor_stats["failed_calls"]:
                            raise RuntimeError("advisor cloud call failed during shaped rollout")
                    if CONFIG.advisor_reward_mode == "counterfactual_necessity" and advisor_stats is not None:
                        if advisor_stats["calls"] != len(advisor_call_output_ids):
                            raise RuntimeError(
                                "advisor audit/captured-call mismatch: "
                                f"audit={advisor_stats['calls']} captured={len(advisor_call_output_ids)}"
                            )
                    if CONFIG.advisor_reward_mode == "counterfactual_cost_budget" and advisor_stats is not None:
                        if any(cost <= 0 for cost in advisor_stats["call_costs_rmb"]):
                            raise RuntimeError("cost-budget rollout has a cloud call without positive cost data")
                    transcript = await _read_transcript(sb)
                    if advisor_stats is not None:
                        advisor_stats.update(
                            model_requests=agent_turn_stats["model_requests"],
                            model_responses=agent_turn_stats["model_responses"],
                            followup_model_requests=agent_turn_stats["advisor_followup_requests"],
                            followup_model_responses=agent_turn_stats["advisor_followup_responses"],
                        )
                        advisor_stats = _reconcile_advisor_transcript(advisor_stats, transcript)
                    reward, advisor_extra = REWARD_MODULE.finalize_rollout(
                        reward_result.score,
                        advisor_stats,
                        penalty_per_call=CONFIG.advisor_penalty_per_call,
                        penalty_cap=CONFIG.advisor_penalty_cap,
                        role=reward_role,
                    )
                finally:
                    if rollout_advisor_enabled and advisor_stats is None:
                        try:
                            advisor_stats = await _collect_advisor_usage(sb)
                        except Exception:
                            logger.warning(
                                "[openclaw_rl] task=%s could not collect advisor audit during cleanup:\n%s",
                                task.task_id,
                                traceback.format_exc(),
                            )

            _timeline_event(timeline, "episode_end", status="completed")
            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=reward,
                extra_metadata={
                    **(base_sample.metadata or {}),
                    "task_id": task.task_id,
                    "agent_exit_code": agent_result.exit_code,
                    "agent_truncated": agent_result.truncated,
                    "agent_process_failure": False,
                    "verifier_failure": reward_result.verifier_failed,
                    "sandbox_elapsed_sec": time.time() - started,
                    "episode_timeline": timeline["events"],
                    "transcript_events": len(transcript),
                    **({"transcript": transcript} if CONFIG.save_transcript else {}),
                    "agent_turns": agent_turns,
                    "agent_tool_calls": agent_tool_calls,
                    "agent_model_requests": agent_turn_stats["model_requests"],
                    "agent_model_responses": agent_turn_stats["model_responses"],
                    "agent_generated_tokens": agent_turn_stats["generated_tokens"],
                    "agent_max_context_tokens": agent_turn_stats["max_context_tokens"],
                    "advisor_followup_model_requests": agent_turn_stats["advisor_followup_requests"],
                    "advisor_followup_model_responses": agent_turn_stats["advisor_followup_responses"],
                    "advisor_force_cloud": CONFIG.advisor_force_cloud,
                    "advisor_style": CONFIG.advisor_style,
                    **advisor_extra,
                },
            )
            if not samples:
                _timeline_event(timeline, "episode_abort", reason="adapter_session_empty")
                return _abort_result(
                    base_sample,
                    "adapter_session_empty",
                    task_id=task_id,
                    reward_role=reward_role,
                    agent_exit_code=agent_result.exit_code,
                    agent_truncated=agent_result.truncated,
                    verifier_failure=reward_result.verifier_failed,
                    advisor_stats=advisor_stats,
                    transcript=transcript if CONFIG.save_transcript else None,
                    episode_timeline=timeline["events"],
                )
            # Mask the dangling turn before advisor-call annotation: the
            # annotation only marks positions that remain trainable, and a cut
            # cloud-call turn must not be treated as attributable spend.
            last_turn_tool_calls, last_turn_output_len = state.pop_last_turn_shape(session_id)
            dangling_masked_tokens = _mask_dangling_final_turn(
                samples,
                last_turn_tool_calls=last_turn_tool_calls,
                last_turn_output_len=last_turn_output_len,
            )
            for masked_sample in samples:
                masked_sample.metadata = {
                    **(masked_sample.metadata or {}),
                    "dangling_final_turn_masked_tokens": dangling_masked_tokens,
                }
            if CONFIG.advisor_reward_mode == "counterfactual_necessity":
                _annotate_advisor_call_tokens(
                    samples,
                    advisor_call_output_ids,
                )
            logger.info(
                "[openclaw_rl] task=%s reward=%.4f exit=%s truncated=%s verifier_failed=%s "
                "samples=%d elapsed=%.1fs advisor_calls=%s",
                task.task_id,
                reward,
                agent_result.exit_code,
                agent_result.truncated,
                reward_result.verifier_failed,
                len(samples),
                time.time() - started,
                advisor_extra.get("advisor_calls", "-"),
            )
            return samples
    except AgentProcessError as exc:
        agent_turn_stats = state.pop_agent_turn_stats(session_id)
        if advisor_stats is not None:
            advisor_stats.update(
                model_requests=agent_turn_stats["model_requests"],
                model_responses=agent_turn_stats["model_responses"],
                followup_model_requests=agent_turn_stats["advisor_followup_requests"],
                followup_model_responses=agent_turn_stats["advisor_followup_responses"],
            )
        logger.warning(
            "[openclaw_rl] task=%s agent process failed exit=%d: %s",
            task.task_id,
            exc.exit_code,
            exc.log[-2000:],
        )
        _timeline_event(timeline, "episode_abort", reason="agent_process_failure", exit_code=exc.exit_code)
        return _abort_result(
            base_sample,
            "agent_process_failure",
            task_id=task_id,
            reward_role=reward_role,
            agent_exit_code=exc.exit_code,
            agent_process_failure=True,
            advisor_stats=advisor_stats,
            agent_turn_stats=agent_turn_stats,
            transcript=transcript_holder.get("transcript"),
            episode_timeline=timeline["events"],
        )
    except asyncio.TimeoutError as exc:
        # asyncio.TimeoutError is the builtin TimeoutError on 3.11+, so an inner
        # step that times out on its own (a gateway that never reports healthy,
        # a sandbox RPC) lands here too. Attributing those to the outer guard
        # discards the only diagnostic the step produced, so separate the two
        # and always log what actually failed.
        elapsed = time.time() - started
        guard_expired = elapsed >= CONFIG.rollout_guard_sec
        logger.warning(
            "[openclaw_rl] task=%s timed out after %.1fs (guard=%ds, guard_expired=%s): %s\n%s",
            task.task_id,
            elapsed,
            CONFIG.rollout_guard_sec,
            guard_expired,
            exc,
            traceback.format_exc(),
        )
        agent_turn_stats = state.pop_agent_turn_stats(session_id)
        if advisor_stats is not None:
            advisor_stats.update(
                model_requests=agent_turn_stats["model_requests"],
                model_responses=agent_turn_stats["model_responses"],
                followup_model_requests=agent_turn_stats["advisor_followup_requests"],
                followup_model_responses=agent_turn_stats["advisor_followup_responses"],
            )
        _timeline_event(
            timeline,
            "episode_abort",
            reason="wall_clock_timeout" if guard_expired else f"step_timeout:{exc}"[:200],
        )
        return _abort_result(
            base_sample,
            "wall_clock_timeout" if guard_expired else f"step_timeout:{exc}"[:200],
            task_id=task_id,
            reward_role=reward_role,
            agent_exit_code=agent_result.exit_code if agent_result is not None else None,
            agent_truncated=agent_result.truncated if agent_result is not None else False,
            verifier_failure=reward_result.verifier_failed if reward_result is not None else False,
            advisor_stats=advisor_stats,
            agent_turn_stats=agent_turn_stats,
            transcript=transcript_holder.get("transcript"),
            episode_timeline=timeline["events"],
        )
    except Exception as exc:
        logger.warning(
            "[openclaw_rl] task=%s failed: %s\n%s",
            task.task_id,
            exc,
            traceback.format_exc(),
        )
        agent_turn_stats = state.pop_agent_turn_stats(session_id)
        if advisor_stats is not None:
            advisor_stats.update(
                model_requests=agent_turn_stats["model_requests"],
                model_responses=agent_turn_stats["model_responses"],
                followup_model_requests=agent_turn_stats["advisor_followup_requests"],
                followup_model_responses=agent_turn_stats["advisor_followup_responses"],
            )
        _timeline_event(timeline, "episode_abort", reason=f"exception:{type(exc).__name__}")
        return _abort_result(
            base_sample,
            f"exception:{type(exc).__name__}",
            task_id=task_id,
            reward_role=reward_role,
            agent_exit_code=agent_result.exit_code if agent_result is not None else None,
            agent_truncated=agent_result.truncated if agent_result is not None else False,
            verifier_failure=reward_result.verifier_failed if reward_result is not None else False,
            advisor_stats=advisor_stats,
            agent_turn_stats=agent_turn_stats,
            transcript=transcript_holder.get("transcript"),
            episode_timeline=timeline["events"],
        )
    finally:
        state.pop_last_message(session_id)
        state.pop_last_turn_shape(session_id)
        state.pop_advisor_call_output_ids(session_id)
        state.pop_agent_turn_stats(session_id)
        state.clear_replay_probe(session_id)
        await state.adapter.drop_session(session_id)


def normalize_trajectory_rewards(args, samples: list[Sample]):
    """Backward-compatible entry point for the preserved static reward."""
    from .rewards.static_penalty import post_process_rewards

    return post_process_rewards(args, samples)


def _rollout_concurrency(args) -> int:
    prompt_concurrency = int(
        getattr(args, "over_sampling_batch_size", None) or getattr(args, "rollout_batch_size", 0)
    )
    samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 0))
    if prompt_concurrency < 1 or samples_per_prompt < 1:
        raise RuntimeError("OpenClaw rollout concurrency requires positive batch size and samples per prompt")
    return prompt_concurrency * samples_per_prompt


def _get_e2b_actor_pool(args) -> RayE2BSandboxPool:
    global _E2B_ACTOR_POOL
    if _E2B_ACTOR_POOL is None:
        rollout_concurrency = _rollout_concurrency(args)
        actor_count = CONFIG.e2b_actor_count or math.ceil(
            rollout_concurrency / CONFIG.e2b_actor_max_concurrency
        )
        logger.info(
            "[openclaw-e2b-actor] rollout_concurrency=%d actors=%d max_concurrency=%d",
            rollout_concurrency,
            actor_count,
            CONFIG.e2b_actor_max_concurrency,
        )
        _E2B_ACTOR_POOL = RayE2BSandboxPool(actor_count, CONFIG.e2b_actor_max_concurrency)
    return _E2B_ACTOR_POOL


def _openclaw_sandbox(args, session_id: str) -> Sandbox:
    custom_config = {
        "image": CONFIG.image,
        "imageRegistryType": CONFIG.image_registry_type,
        "command": ["/bin/sh", "-c"],
        "args": ["/mnt/envd -port 49983"],
        "resources": {"cpu": CONFIG.cpu, "memory": CONFIG.memory},
    }
    return _get_e2b_actor_pool(args).sandbox(
        session_id,
        json.dumps(custom_config),
        timeout=CONFIG.rollout_guard_sec + 300,
        image_metadata_key="x-custom-config",
        template=CONFIG.template,
        strict_read=True,
    )


async def _wait_sandbox_ready(sb: Sandbox) -> None:
    """Wait for envd command execution before starting task initialization."""
    deadline = time.monotonic() + 60
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ec, _, stderr = await sb.exec("true", user="root", timeout=10, idempotent=True)
            if ec == 0:
                return
            last_error = RuntimeError(f"sandbox readiness command exited {ec}: {stderr[-400:]}")
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(2)
    raise TimeoutError(f"E2B sandbox did not become ready: {last_error}") from last_error


@asynccontextmanager
async def _prepared_rollout_sandbox(
    args,
    task: ClawGymTask,
    session_id: str,
    adapter_url: str,
    *,
    advisor_enabled: bool,
    timeline: dict[str, Any] | None = None,
    transcript_holder: dict[str, list[dict[str, Any]]] | None = None,
):
    """Create and fully initialize a clean sandbox before agent execution.

    A setup failure discards the entire sandbox and starts from a new one. Once
    this context yields, the caller may launch the agent and no setup retry can
    replay rollout operations.
    """
    manager = None
    sb = None
    for attempt in range(CONFIG.setup_attempts):
        _timeline_event(timeline, "sandbox_create_start", setup_attempt=attempt + 1)
        candidate = _openclaw_sandbox(args, session_id)
        try:
            entered = await candidate.__aenter__()
        except BaseException as exc:
            await candidate.__aexit__(type(exc), exc, exc.__traceback__)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                "E2B create exhausted its RPC retry budget; setup was not started"
            ) from exc

        try:
            await _wait_sandbox_ready(entered)
            _timeline_event(timeline, "sandbox_ready", setup_attempt=attempt + 1)
            await _prepare_workspace(entered, task)
            await _configure_openclaw(entered, session_id, adapter_url)
            await _configure_search(entered)
            if advisor_enabled:
                await _configure_advisor(entered, task, session_id)
            await _start_gateway(
                entered,
                session_id,
                advisor_enabled=advisor_enabled,
                timeline=timeline,
            )
            await _prepare_agent(entered, task, session_id, advisor_enabled=advisor_enabled)
            _timeline_event(timeline, "agent_prepared")
        except BaseException as exc:
            await candidate.__aexit__(type(exc), exc, exc.__traceback__)
            if not isinstance(exc, Exception):
                raise
            if attempt + 1 >= CONFIG.setup_attempts:
                raise RuntimeError(
                    f"E2B setup failed after {CONFIG.setup_attempts} clean sandbox attempts"
                ) from exc
            backoff = CONFIG.setup_backoff_base_sec * (2**attempt) * random.uniform(0.5, 1.5)
            logger.warning(
                "[openclaw_rl] setup attempt %d/%d failed for task=%s; recreating in %.1fs: %s",
                attempt + 1,
                CONFIG.setup_attempts,
                task.task_id,
                backoff,
                exc,
            )
            await asyncio.sleep(backoff)
            continue
        manager = candidate
        sb = entered
        break

    assert manager is not None and sb is not None
    rollout_cancelled = False
    try:
        _timeline_event(timeline, "sandbox_enter")
        yield sb
    except asyncio.CancelledError:
        # Cancellation can be delivered at the yield point and then be
        # consumed before diagnostics starts; retain the signal explicitly so
        # cleanup does not issue new RPCs on an aborted rollout.
        rollout_cancelled = True
        raise
    finally:
        if CONFIG.save_transcript and transcript_holder is not None:
            try:
                transcript_holder["transcript"] = await _read_transcript(sb)
            except Exception:
                logger.warning(
                    "[openclaw_rl] transcript capture failed during cleanup task=%s:\n%s",
                    task.task_id,
                    traceback.format_exc(),
                )
        if CONFIG.runtime_diagnostics_enabled:
            _timeline_event(timeline, "runtime_diagnostics_start")
            diagnostics_task = asyncio.create_task(
                _collect_runtime_diagnostics(sb, task.task_id, timeline=timeline)
            )
        else:
            _timeline_event(timeline, "runtime_diagnostics_skipped", reason="disabled")
            diagnostics_task = asyncio.create_task(asyncio.sleep(0))
        diagnostics_started = time.monotonic()
        diagnostics_timed_out = False
        cleanup_cancelled = rollout_cancelled

        def consume_diagnostics_result(done_task: asyncio.Task) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "[openclaw_rl] runtime diagnostics task failed after cancellation task=%s:\n%s",
                    task.task_id,
                    traceback.format_exc(),
                )

        def diagnostics_remaining() -> float:
            return max(
                0.0,
                CONFIG.runtime_diagnostics_timeout_sec - (time.monotonic() - diagnostics_started),
            )

        def cancel_diagnostics(reason: str) -> None:
            nonlocal diagnostics_timed_out
            diagnostics_timed_out = True
            _timeline_event(
                timeline,
                "runtime_diagnostics_timeout",
                timeout_sec=CONFIG.runtime_diagnostics_timeout_sec,
                reason=reason,
            )
            logger.warning(
                "[openclaw_rl] runtime diagnostics exceeded %.1fs during %s task=%s; "
                "continuing cleanup",
                CONFIG.runtime_diagnostics_timeout_sec,
                reason,
                task.task_id,
            )
            if not diagnostics_task.done():
                diagnostics_task.cancel()
                # Do not await cancellation: an E2B RPC may itself ignore
                # cancellation. The callback prevents an eventual exception
                # from becoming an unhandled-task warning.
                diagnostics_task.add_done_callback(consume_diagnostics_result)

        def consume_remote_cleanup_result(done_task: asyncio.Task, operation: str) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "[openclaw_rl] remote cleanup task failed after timeout "
                    "operation=%s task=%s:\n%s",
                    operation,
                    task.task_id,
                    traceback.format_exc(),
                )

        async def bounded_remote_cleanup(
            operation: str,
            operation_factory,
            deadline: float,
        ) -> bool:
            """Run one remote cleanup RPC without extending the cleanup budget.

            The operation is shielded so wait_for does not wait for a transport
            that ignores cancellation. On timeout, cancel and detach the task;
            the E2B sandbox lifetime remains the final remote cleanup fallback.
            """
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                _timeline_event(timeline, "remote_cleanup_skipped", operation=operation, reason="budget_exhausted")
                logger.warning(
                    "[openclaw_rl] remote cleanup budget exhausted; skipping %s task=%s",
                    operation,
                    task.task_id,
                )
                return False

            cleanup_task = asyncio.create_task(operation_factory())
            try:
                await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=remaining)
            except asyncio.TimeoutError:
                _timeline_event(
                    timeline,
                    "remote_cleanup_timeout",
                    operation=operation,
                    budget_sec=remaining,
                )
                logger.warning(
                    "[openclaw_rl] remote cleanup operation exceeded %.1fs; "
                    "detaching %s task=%s",
                    remaining,
                    operation,
                    task.task_id,
                )
                if not cleanup_task.done():
                    cleanup_task.cancel()
                    cleanup_task.add_done_callback(
                        lambda done_task: consume_remote_cleanup_result(done_task, operation)
                    )
                return False
            except asyncio.CancelledError:
                if not cleanup_task.done():
                    cleanup_task.cancel()
                    cleanup_task.add_done_callback(
                        lambda done_task: consume_remote_cleanup_result(done_task, operation)
                    )
                raise
            except Exception as exc:
                _timeline_event(
                    timeline,
                    "remote_cleanup_failed",
                    operation=operation,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
                logger.warning(
                    "[openclaw_rl] remote cleanup failed operation=%s task=%s: %s",
                    operation,
                    task.task_id,
                    exc,
                )
                return False
            _timeline_event(timeline, "remote_cleanup_completed", operation=operation)
            return True

        try:
            await asyncio.wait_for(
                asyncio.shield(diagnostics_task),
                timeout=diagnostics_remaining(),
            )
        except asyncio.TimeoutError:
            cancel_diagnostics("diagnostics_timeout")
        except asyncio.CancelledError:
            cleanup_cancelled = True
            # Preserve diagnostics for a bounded period when the outer rollout
            # guard fires, then let the original cancellation reach the abort
            # handler instead of blocking the episode indefinitely.
            try:
                remaining = diagnostics_remaining()
                if remaining > 0:
                    await asyncio.wait_for(asyncio.shield(diagnostics_task), timeout=remaining)
                else:
                    cancel_diagnostics("outer_rollout_cancelled")
            except asyncio.TimeoutError:
                cancel_diagnostics("outer_rollout_cancelled")
            except Exception:
                logger.warning(
                    "[openclaw_rl] runtime diagnostics failed during cancellation task=%s:\n%s",
                    task.task_id,
                    traceback.format_exc(),
                )
        except Exception:
            logger.warning(
                "[openclaw_rl] runtime diagnostics failed task=%s:\n%s",
                task.task_id,
                traceback.format_exc(),
            )
        if CONFIG.runtime_diagnostics_enabled:
            _timeline_event(
                timeline,
                "runtime_diagnostics_end",
                completed=not diagnostics_timed_out,
            )
        skip_remote_cleanup = diagnostics_timed_out or cleanup_cancelled
        if skip_remote_cleanup:
            reason = "diagnostics_timeout" if diagnostics_timed_out else "outer_rollout_cancelled"
            _timeline_event(timeline, "remote_cleanup_skipped", reason=reason)
            logger.warning(
                "[openclaw_rl] skipping gateway/sandbox teardown on dead or cancelled "
                "transport task=%s reason=%s; E2B lifetime will reclaim the sandbox",
                task.task_id,
                reason,
            )
        else:
            cleanup_deadline = time.monotonic() + CONFIG.remote_cleanup_timeout_sec
            try:
                gateway_stopped = await bounded_remote_cleanup(
                    "gateway_stop",
                    lambda: _stop_gateway(sb),
                    cleanup_deadline,
                )
            except asyncio.CancelledError:
                cleanup_cancelled = True
                gateway_stopped = False
            if gateway_stopped:
                _timeline_event(timeline, "gateway_stop_requested")

            if not cleanup_cancelled:
                try:
                    await bounded_remote_cleanup(
                        "sandbox_teardown",
                        lambda: manager.__aexit__(None, None, None),
                        cleanup_deadline,
                    )
                except asyncio.CancelledError:
                    cleanup_cancelled = True
        if cleanup_cancelled:
            raise asyncio.CancelledError


async def _prepare_workspace(sb: Sandbox, task: ClawGymTask) -> None:
    await sb.exec(
        f"rm -rf {WORKSPACE} /tmp_workspace && mkdir -p /tmp_workspace && "
        f"mkdir -p {WORKSPACE.parent} && ln -s /tmp_workspace {WORKSPACE}",
        user="root",
        check=True,
        idempotent=True,
    )
    if task.input_files_dir.is_dir():
        relative_mount = task.input_mount_dir.relative_to(WORKSPACE)
        destination = WORKSPACE / relative_mount
        await _upload_tree(sb, task.input_files_dir, destination)


async def _configure_openclaw(sb: Sandbox, session_id: str, adapter_url: str) -> None:
    provider = {
        "providers": {
            "slime": {
                "baseUrl": _openai_base_url(adapter_url),
                "apiKey": session_id,
                "api": "openai-completions",
                "models": [
                    {
                        "id": "slime-actor",
                        "name": "Slime Actor",
                        "reasoning": CONFIG.enable_thinking,
                        "input": ["text"],
                        "contextWindow": CONFIG.context_window,
                        "maxTokens": CONFIG.max_tokens,
                        "compat": {"supportsUsageInStreaming": True, "supportsTools": True},
                    }
                ],
            }
        }
    }
    await sb.write_file("/tmp/slime_models.json", json.dumps(provider), user="root")
    patch_script = f"""import json
from pathlib import Path
p = Path('/root/.openclaw/openclaw.json')
c = json.loads(p.read_text()) if p.exists() else {{}}
c['models'] = json.loads(Path('/tmp/slime_models.json').read_text())
defaults = c.setdefault('agents', {{}}).setdefault('defaults', {{}})
defaults['model'] = {{'primary': 'slime/slime-actor'}}
defaults['workspace'] = {str(WORKSPACE)!r}
defaults['timeoutSeconds'] = {CONFIG.agent_timeout_sec}
tools = c.setdefault('tools', {{}})
tools['profile'] = 'coding'
deny = tools.setdefault('deny', [])
for name in ('web_search', 'web_fetch'):
    while name in deny:
        deny.remove(name)
if 'memory_search' not in deny:
    deny.append('memory_search')
allow_key = 'allow' if 'allow' in tools else 'alsoAllow'
allow = tools.setdefault(allow_key, [])
for name in ('web_search', 'web_fetch'):
    if '*' not in allow and name not in allow:
        allow.append(name)
if allow_key == 'allow':
    tools.pop('alsoAllow', None)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(c, indent=2))
"""
    await sb.write_file("/tmp/configure_slime_openclaw.py", patch_script, user="root")
    await sb.exec("python3 /tmp/configure_slime_openclaw.py", user="root", check=True, idempotent=True)
    await sb.exec(
        "openclaw models set 'slime/slime-actor'",
        user="root",
        check=True,
        timeout=60,
        idempotent=True,
    )


async def _disable_image_web_tools(sb: Sandbox) -> None:
    """Drop the image's built-in web tool config, including its env references."""
    patch_script = """import json
from pathlib import Path
p = Path('/root/.openclaw/openclaw.json')
c = json.loads(p.read_text()) if p.exists() else {}
web = c.setdefault('tools', {}).setdefault('web', {})
web['search'] = {'enabled': False}
web['fetch'] = {'enabled': False}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(c, indent=2))
"""
    await sb.write_file("/tmp/disable_web_tools.py", patch_script, user="root")
    await sb.exec(
        "python3 /tmp/disable_web_tools.py", user="root", check=True, timeout=30, idempotent=True
    )


async def _configure_search(sb: Sandbox) -> None:
    """Install the configured search and reader backends into the sandbox."""
    if CONFIG.search_provider in {"none", "disabled", "off"}:
        # The image ships an openclaw.json whose tools.web.search.apiKey reads
        # ${BRAVE_API_KEY}. Leaving it in place makes the gateway refuse to come
        # up, so replace the web tool config here too — the serp branch below
        # gets this for free by overwriting the same keys.
        await _disable_image_web_tools(sb)
        return
    if not CONFIG.search_api_key:
        raise RuntimeError("SEARCH_PROVIDER=serp requires the configured search API key")

    plugins = {
        "serp-search": Path(CONFIG.search_plugin_dir),
        "uda-reader": Path(CONFIG.reader_plugin_dir),
    }
    for plugin_id, plugin_src in plugins.items():
        if not plugin_src.is_dir():
            raise RuntimeError(f"{plugin_id} plugin dir not found: {plugin_src}")
        dst = PurePosixPath("/tmp") / plugin_id
        await sb.exec(
            f"rm -rf {shlex.quote(str(dst))}", user="root", check=True, timeout=30, idempotent=True
        )
        await _upload_tree(sb, plugin_src, dst)
        await sb.exec(
            f"openclaw plugins install {shlex.quote(str(dst))}",
            user="root",
            check=True,
            timeout=600,
            idempotent=False,
        )

    patch_script = """import json
from pathlib import Path
p = Path('/root/.openclaw/openclaw.json')
c = json.loads(p.read_text()) if p.exists() else {}
web = c.setdefault('tools', {}).setdefault('web', {})
web['search'] = {'enabled': False}
web['fetch'] = {'enabled': False}
entries = c.setdefault('plugins', {}).setdefault('entries', {})
entries.setdefault('serp-search', {})['enabled'] = True
entries.setdefault('uda-reader', {})['enabled'] = True
p.write_text(json.dumps(c, indent=2))
"""
    await sb.write_file("/tmp/configure_search.py", patch_script, user="root")
    await sb.exec("python3 /tmp/configure_search.py", user="root", check=True, timeout=30, idempotent=True)


def _openai_base_url(url: str) -> str:
    """Return a complete OpenAI-compatible base URL ending in /v1."""
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _agent_message(task: ClawGymTask, *, advisor_enabled: bool) -> str:
    """The user message handed to the OpenClaw agent, with the advisor guidance
    appended when advisor mode is on."""
    if advisor_enabled:
        return task.user_query + REWARD_MODULE.advisor_prompt(ADVISOR_PROMPT)
    return task.user_query


def _advisor_env_exports(session_id: str, *, advisor_enabled: bool) -> str:
    """Bash `export` lines carrying the cloud-assist plugin's runtime config.

    Emitted (identically) for BOTH the gateway and the agent launch: OpenClaw may
    execute plugin tools in either process, and each OS process only sees the env
    it was started with. Mirroring the eval side (which sources these before every
    command) guarantees `cloud_assist` finds CLOUD_* and the transcript no matter
    which process runs it. Empty string when advisor mode is off.
    """
    if not advisor_enabled:
        return ""
    # Point the plugin straight at this session's transcript (main agent + our
    # session id) so it never falls back to an empty context, and hand it the
    # fixed cloud endpoint.
    chat_jsonl = f"/root/.openclaw/agents/main/sessions/{session_id}.jsonl"
    advisor_env = {
        "HOME": "/root",
        "CLOUD_API_KEY": CONFIG.cloud_api_key,
        "CLOUD_BASE_URL": CONFIG.cloud_base_url,
        "CLOUD_MODEL": CONFIG.cloud_model,
        "ADVISOR_STYLE": CONFIG.advisor_style,
        "ADVISOR_CHAT_JSONL": chat_jsonl,
        "ADVISOR_AGENT_ID": "main",
        "ADVISOR_SESSION_ID": session_id,
        "ADVISOR_CTX_PATH": ADVISOR_CTX_PATH,
        "ADVISOR_AUDIT_LOG": ADVISOR_AUDIT_PATH,
        "ADVISOR_TRACE_LOG": ADVISOR_TRACE_PATH,
        "CLOUD_MAX_ATTEMPTS": "3",
        "CLOUD_REQUEST_TIMEOUT_MS": "240000",
    }
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in advisor_env.items() if value)


def _web_tool_env_exports() -> str:
    env = {
        "SERP_API_KEY": CONFIG.search_api_key,
        "UDA_READER_ENDPOINT": CONFIG.reader_endpoint,
        "UDA_READER_API_KEY": CONFIG.reader_api_key,
        "UDA_READER_WORK_PROJECT_ID": CONFIG.reader_work_project_id,
        "UDA_READER_PROXY_PROJECT": CONFIG.reader_proxy_project,
        "UDA_READER_PROXY_SOURCES": CONFIG.reader_proxy_sources,
    }
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in env.items() if key and value)


async def _configure_advisor(sb: Sandbox, task: ClawGymTask, session_id: str) -> None:
    """Install the cloud-assist-advisor plugin and allow its `cloud_assist` tool.

    The plugin registers a parameterless tool that forwards the edge actor's
    transcript to a fixed external stronger model (CLOUD_* env, set at agent-run
    time). Its advice returns as a tool result, so it lands in the next prompt as
    loss_mask=0 context and is never trained — only the edge model's decision to
    call it and its follow-up actions carry gradient.
    """
    plugin_src = Path(CONFIG.advisor_plugin_dir)
    if not plugin_src.is_dir():
        raise RuntimeError(f"advisor plugin dir not found: {plugin_src}")
    dst = PurePosixPath("/tmp") / plugin_src.name
    await sb.exec(
        f"rm -rf {shlex.quote(str(dst))}", user="root", check=True, timeout=30, idempotent=True
    )
    await _upload_tree(sb, plugin_src, dst)
    await sb.exec(
        f"openclaw plugins install {shlex.quote(str(dst))}",
        user="root",
        check=True,
        timeout=600,
        idempotent=False,
    )
    patch_script = """import json
from pathlib import Path
p = Path('/root/.openclaw/openclaw.json')
c = json.loads(p.read_text()) if p.exists() else {}
tools = c.setdefault('tools', {})
if 'allow' in tools:
    allow = tools.setdefault('allow', [])
    if '*' not in allow and 'cloud_assist' not in allow:
        allow.append('cloud_assist')
    tools.pop('alsoAllow', None)
else:
    allow = tools.setdefault('alsoAllow', [])
    if 'cloud_assist' not in allow:
        allow.append('cloud_assist')
p.write_text(json.dumps(c, indent=2))
"""
    await sb.write_file("/tmp/configure_advisor.py", patch_script, user="root")
    await sb.exec("python3 /tmp/configure_advisor.py", user="root", check=True, idempotent=True)
    # Let the advisor see the executor's actual instructions (query + guidance).
    await sb.write_file(ADVISOR_CTX_PATH, _agent_message(task, advisor_enabled=True), user="root")


async def _collect_advisor_usage(sb: Sandbox) -> dict[str, Any]:
    """Aggregate audit usage and recompute per-call cost with configured prices."""
    calls = 0
    successful_calls = 0
    failed_calls = 0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    cache_read_tokens = 0
    cloud_attempts = 0
    cloud_retries = 0
    cloud_latency_ms = 0.0
    cloud_max_latency_ms = 0.0
    failure_status_counts: dict[str, int] = defaultdict(int)
    call_costs_rmb: list[float] = []
    cloud_models: set[str] = set()
    trace_by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    _, raw, _ = await sb.exec(
        f"cat {shlex.quote(ADVISOR_AUDIT_PATH)} 2>/dev/null",
        user="root",
        timeout=30,
        idempotent=True,
    )
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("tool") != "cloud_assist":
            continue
        calls += 1
        model = str(entry.get("model") or "")
        if model:
            cloud_models.add(model)
        status_ok = entry.get("status") == "ok"
        if status_ok:
            successful_calls += 1
        else:
            failed_calls += 1
            failure_status_counts[_advisor_failure_status(entry)] += 1
        try:
            attempts = max(1, int(entry.get("attempts") or 1))
        except (TypeError, ValueError):
            attempts = 1
        try:
            latency_ms = max(0.0, float(entry.get("latency_ms") or 0.0))
        except (TypeError, ValueError):
            latency_ms = 0.0
        cloud_attempts += attempts
        cloud_retries += attempts - 1
        cloud_latency_ms += latency_ms
        cloud_max_latency_ms = max(cloud_max_latency_ms, latency_ms)
        call_prompt = int(entry.get("prompt_tokens") or 0)
        call_completion = int(entry.get("completion_tokens") or 0)
        call_cache_read = min(call_prompt, int(entry.get("cache_read_tokens") or 0))
        call_total = int(entry.get("total_tokens") or call_prompt + call_completion)
        call_cost = (
            (call_prompt - call_cache_read) * CONFIG.advisor_cost_noncached_input_per_1k_rmb
            + call_cache_read * CONFIG.advisor_cost_cached_input_per_1k_rmb
            + call_completion * CONFIG.advisor_cost_output_per_1k_rmb
        ) / 1000.0
        prompt_tokens += call_prompt
        completion_tokens += call_completion
        cache_read_tokens += call_cache_read
        total_tokens += call_total
        call_costs_rmb.append(call_cost)
    _, trace_raw, _ = await sb.exec(
        f"cat {shlex.quote(ADVISOR_TRACE_PATH)} 2>/dev/null",
        user="root",
        timeout=30,
        idempotent=True,
    )
    trace_attempts = 0
    trace_attempt_errors = 0
    trace_attempt_timeouts = 0
    trace_returned_call_ids: list[str] = []
    for line in trace_raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("tool") != "cloud_assist":
            continue
        call_id = str(event.get("call_id") or "")
        if call_id:
            trace_by_call[call_id].append(event)
        event_name = event.get("event")
        if event_name == "attempt_start":
            trace_attempts += 1
        elif event_name == "attempt_error":
            trace_attempt_errors += 1
            error_text = f"{event.get('error_name', '')} {event.get('error_message', '')}".lower()
            if "timeout" in error_text or "aborted" in error_text:
                trace_attempt_timeouts += 1
        elif event_name == "tool_execute_return" and call_id:
            trace_returned_call_ids.append(call_id)
    tool_invocations = 0
    tool_returns = 0
    cloud_responses = 0
    pending_tool_calls = 0
    cloud_response_without_tool_return = 0
    for events in trace_by_call.values():
        names = {str(event.get("event") or "") for event in events}
        invoked = "tool_execute_start" in names
        returned = "tool_execute_return" in names
        responded = "response_parsed" in names or "cloud_call_success" in names
        tool_invocations += int(invoked)
        tool_returns += int(returned)
        cloud_responses += int(responded)
        pending_tool_calls += int(invoked and not returned)
        cloud_response_without_tool_return += int(responded and not returned)
    if len(cloud_models) > 1:
        raise RuntimeError(f"advisor audit mixes cloud models: {sorted(cloud_models)}")
    cloud_model = next(iter(cloud_models), "")
    if cloud_model and cloud_model != CONFIG.cloud_model:
        raise RuntimeError(f"advisor audit model {cloud_model!r} != configured CLOUD_MODEL {CONFIG.cloud_model!r}")
    pricing = {
        "noncached_input": CONFIG.advisor_cost_noncached_input_per_1k_rmb,
        "cached_input": CONFIG.advisor_cost_cached_input_per_1k_rmb,
        "output": CONFIG.advisor_cost_output_per_1k_rmb,
    }
    return {
        "calls": calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "cloud_attempts": cloud_attempts,
        "cloud_retries": cloud_retries,
        "cloud_latency_ms": cloud_latency_ms,
        "cloud_max_latency_ms": cloud_max_latency_ms,
        "failure_status_counts": dict(failure_status_counts),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "call_costs_rmb": call_costs_rmb,
        "cost_rmb": sum(call_costs_rmb),
        "cloud_model": cloud_model or CONFIG.cloud_model,
        "pricing_rmb_per_1k": pricing,
        "tool_invocations": tool_invocations,
        "tool_returns": tool_returns,
        "pending_tool_calls": pending_tool_calls,
        "cloud_responses": cloud_responses,
        "cloud_response_without_tool_return": cloud_response_without_tool_return,
        "trace_attempts": trace_attempts,
        "trace_attempt_errors": trace_attempt_errors,
        "trace_attempt_timeouts": trace_attempt_timeouts,
        "trace_returned_call_ids": sorted(set(trace_returned_call_ids)),
    }


def _reconcile_advisor_transcript(
    advisor_stats: dict[str, Any], transcript: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cross-check plugin returns against tool results durably stored by OpenClaw."""
    tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    committed_call_ids: set[str] = set()
    result_waiting_for_assistant = 0
    tool_results_without_followup = 0
    for event in transcript:
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "toolCall"
                    and part.get("name") == "cloud_assist"
                    and part.get("id")
                ):
                    tool_call_ids.add(str(part["id"]))
        if message.get("role") == "toolResult" and message.get("toolName") == "cloud_assist":
            tool_call_id = message.get("toolCallId")
            if tool_call_id:
                tool_result_ids.add(str(tool_call_id))
            details = message.get("details") or {}
            if isinstance(details, dict) and details.get("callId"):
                committed_call_ids.add(str(details["callId"]))
            result_waiting_for_assistant += 1
        elif message.get("role") == "assistant" and result_waiting_for_assistant:
            result_waiting_for_assistant = 0
    tool_results_without_followup = result_waiting_for_assistant
    returned_call_ids = set(advisor_stats.pop("trace_returned_call_ids", []))
    advisor_stats.update(
        transcript_tool_calls=len(tool_call_ids),
        transcript_tool_results=len(tool_result_ids),
        transcript_pending_tool_calls=len(tool_call_ids - tool_result_ids),
        committed_tool_result_calls=len(committed_call_ids),
        tool_return_without_committed_result=len(returned_call_ids - committed_call_ids),
        transcript_tool_results_without_followup=tool_results_without_followup,
    )
    return advisor_stats


def _advisor_failure_status(entry: dict[str, Any]) -> str:
    try:
        http_status = int(entry.get("http_status") or 0)
    except (TypeError, ValueError):
        http_status = 0
    if http_status > 0:
        return f"http_{http_status}"
    cause_code = re.sub(r"[^a-z0-9]+", "_", str(entry.get("error_cause_code") or "").lower()).strip("_")
    if cause_code:
        return f"network_{cause_code}"
    error_name = re.sub(r"[^a-z0-9]+", "_", str(entry.get("error_name") or "").lower()).strip("_")
    if error_name and error_name != "error":
        return f"error_{error_name}"
    return "unknown"


async def _start_gateway(
    sb: Sandbox,
    session_id: str,
    *,
    advisor_enabled: bool,
    timeline: dict[str, Any] | None = None,
) -> None:
    _timeline_event(timeline, "gateway_start_command", port=CONFIG.gateway_port)
    await sb.exec(
        "rm -rf /tmp/openclaw_gateway.spawned && "
        "rm -f /tmp/openclaw_gateway.pid /tmp/openclaw_gateway.log",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )
    launch = (
        "if mkdir /tmp/openclaw_gateway.spawned 2>/dev/null; then\n"
        f"{_web_tool_env_exports()}"
        f"{_advisor_env_exports(session_id, advisor_enabled=advisor_enabled)}"
        f"setsid openclaw gateway --port {CONFIG.gateway_port} "
        "> /tmp/openclaw_gateway.log 2>&1 < /dev/null &\n"
        "echo $! > /tmp/openclaw_gateway.pid\n"
        "fi"
    )
    await sb.exec(
        f"bash -c {shlex.quote(launch)}",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        ec, _, _ = await sb.exec(
            f"curl -fsS http://127.0.0.1:{CONFIG.gateway_port}/health >/dev/null",
            user="root",
            timeout=10,
            idempotent=True,
        )
        if ec == 0:
            _timeline_event(timeline, "gateway_healthy", port=CONFIG.gateway_port)
            return
        await asyncio.sleep(2)
    _, log, _ = await sb.exec("tail -100 /tmp/openclaw_gateway.log", user="root", idempotent=True)
    _timeline_event(timeline, "gateway_health_timeout", log_tail=log[-2000:])
    raise TimeoutError(f"OpenClaw gateway did not become ready: {log[-2000:]}")


async def _stop_gateway(sb: Sandbox) -> None:
    await sb.exec(
        "test ! -f /tmp/openclaw_gateway.pid || kill -- -$(cat /tmp/openclaw_gateway.pid) 2>/dev/null || true",
        user="root",
        timeout=15,
        idempotent=True,
    )


async def _collect_runtime_diagnostics(
    sb: Sandbox,
    task_id: str,
    *,
    timeline: dict[str, Any] | None = None,
) -> None:
    """Capture process, gateway, agent, and version state before sandbox teardown."""
    max_chars = max(CONFIG.agent_log_tail_chars, 12000)
    commands = {
        "agent_log": f"tail -c {max_chars} /tmp/openclaw_agent.log 2>/dev/null || true",
        "gateway_log": f"tail -c {max_chars} /tmp/openclaw_gateway.log 2>/dev/null || true",
        "agent_done": "cat /tmp/openclaw_agent.done 2>/dev/null || true",
        "agent_pid": "cat /tmp/openclaw_agent.pid 2>/dev/null || true",
        "gateway_pid": "cat /tmp/openclaw_gateway.pid 2>/dev/null || true",
        "openclaw_version": "openclaw --version 2>&1",
        "processes": "ps -ef 2>&1",
        "gateway_health": f"curl -sS --max-time 10 http://127.0.0.1:{CONFIG.gateway_port}/health 2>&1",
    }
    diagnostics: dict[str, Any] = {}

    async def capture(name: str, command: str) -> tuple[str, dict[str, Any]]:
        try:
            ec, stdout, stderr = await sb.exec(
                command,
                user="root",
                timeout=20,
                idempotent=True,
            )
            return name, {
                "exit_code": ec,
                "stdout": stdout[-max_chars:],
                "stderr": stderr[-2000:],
            }
        except Exception as exc:
            return name, {"error": f"{type(exc).__name__}: {exc}"}

    captured = await asyncio.gather(*(capture(name, command) for name, command in commands.items()))
    diagnostics.update(captured)
    record = {
        "task_id": task_id,
        "config": {
            "e2b_template": CONFIG.template,
            "e2b_image": CONFIG.image,
            "gateway_port": CONFIG.gateway_port,
            "agent_timeout_sec": CONFIG.agent_timeout_sec,
            "rollout_guard_sec": CONFIG.rollout_guard_sec,
            "configured_agent_log_tail_chars": CONFIG.agent_log_tail_chars,
            "runtime_diagnostics_timeout_sec": CONFIG.runtime_diagnostics_timeout_sec,
            "remote_cleanup_timeout_sec": CONFIG.remote_cleanup_timeout_sec,
        },
        "agent_log_tail_chars": max_chars,
        "diagnostics": diagnostics,
    }
    logger.info("[openclaw_rl] runtime_diagnostics %s", json.dumps(record, sort_keys=True, default=str))
    _timeline_event(
        timeline,
        "runtime_diagnostics_captured",
        files=list(commands),
        agent_log_tail_chars=max_chars,
    )


async def _prepare_agent(sb: Sandbox, task: ClawGymTask, session_id: str, *, advisor_enabled: bool) -> None:
    """Write the agent launcher while setup is still safe to restart."""
    message = _agent_message(task, advisor_enabled=advisor_enabled)
    env_exports = _web_tool_env_exports() + _advisor_env_exports(session_id, advisor_enabled=advisor_enabled)
    script = (
        "#!/usr/bin/env bash\n"
        f"{env_exports}"
        f"cd {WORKSPACE}\n"
        f"openclaw agent --session-id {shlex.quote(session_id)} "
        f"--timeout {CONFIG.agent_timeout_sec} --message {shlex.quote(message)} "
        "> /tmp/openclaw_agent.log 2>&1\n"
        "echo $? > /tmp/openclaw_agent.done\n"
    )
    await sb.write_file("/tmp/run_openclaw_agent.sh", script, user="root")
    await sb.exec(
        "rm -rf /tmp/openclaw_agent.spawned && "
        "rm -f /tmp/openclaw_agent.pid /tmp/openclaw_agent.done /tmp/openclaw_agent.log && "
        "chmod +x /tmp/run_openclaw_agent.sh",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )


async def _run_agent(
    sb: Sandbox,
    *,
    timeline: dict[str, Any] | None = None,
) -> AgentRunResult:
    """Launch exactly one agent execution and wait for its terminal status."""
    launch = (
        "if mkdir /tmp/openclaw_agent.spawned 2>/dev/null; then "
        "setsid /tmp/run_openclaw_agent.sh < /dev/null > /dev/null 2>&1 & "
        "echo $! > /tmp/openclaw_agent.pid; "
        "fi"
    )
    await sb.exec(
        f"bash -c {shlex.quote(launch)}",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )
    _timeline_event(timeline, "agent_process_started")
    deadline = time.time() + CONFIG.agent_timeout_sec + 60
    while time.time() < deadline:
        ec, out, _ = await sb.exec(
            "test -f /tmp/openclaw_agent.done && cat /tmp/openclaw_agent.done",
            user="root",
            timeout=15,
            idempotent=True,
        )
        if ec == 0 and out.strip():
            try:
                exit_code = int(out.strip())
            except ValueError as exc:
                raise RuntimeError(f"invalid OpenClaw agent exit status: {out!r}") from exc
            agent_log = await sb.read_file("/tmp/openclaw_agent.log", user="root")
            if exit_code != 0:
                raise AgentProcessError(exit_code, agent_log)
            # exit=0 只说明 openclaw 进程正常返回, 不说明 episode 是自然收尾的: 训练里
            # 92% 的轨迹停在一个已发出、未执行完的 tool_call 上(评测侧 100% 以纯文本收尾)。
            # openclaw 把 run 的真实终止原因写在自己的 log 里, 而这里过去直接丢弃了它,
            # 所以 500 步训练一条证据都没留下。保留尾部, 供定位"为什么没到超时就停"。
            if CONFIG.agent_log_tail_chars > 0:
                logger.info(
                    "[openclaw_rl] agent exit=0 log_tail: %s",
                    agent_log[-CONFIG.agent_log_tail_chars :],
                )
            return AgentRunResult(exit_code=exit_code, log=agent_log, truncated=False)
        await asyncio.sleep(5)
    await sb.exec(
        "test ! -f /tmp/openclaw_agent.pid || kill -- -$(cat /tmp/openclaw_agent.pid) 2>/dev/null || true",
        user="root",
        timeout=15,
        idempotent=True,
    )
    agent_log = await sb.read_file("/tmp/openclaw_agent.log", user="root")
    _, advisor_trace, _ = await sb.exec(
        f"tail -50 {shlex.quote(ADVISOR_TRACE_PATH)} 2>/dev/null",
        user="root",
        timeout=15,
        idempotent=True,
    )
    logger.warning("[openclaw_rl] agent reached its deadline and was truncated: %s", agent_log[-2000:])
    _timeline_event(timeline, "agent_deadline", log_tail=agent_log[-2000:])
    if advisor_trace.strip():
        logger.warning("[openclaw_rl] advisor trace at agent deadline:\n%s", advisor_trace)
    return AgentRunResult(exit_code=None, log=agent_log, truncated=True)


async def _run_reward(sb: Sandbox, task: ClawGymTask, final_message: str) -> RewardRunResult:
    await _upload_tree(sb, task.reward_dir, WORKSPACE / "reward")
    await sb.exec(
        f"test -d {shlex.quote(str(WORKSPACE))} && "
        f"test -f {shlex.quote(str(WORKSPACE / 'reward' / 'reward.sh'))}",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )
    transcript = await _read_transcript(sb)
    payload = {
        "task_id": task.task_id,
        "user_query": task.user_query,
        "metadata": task.metadata,
        "final_message": final_message,
        "transcript": transcript,
    }
    payload_path = WORKSPACE / ".openclaw_reward_payload.json"
    await sb.write_file(str(payload_path), json.dumps(payload, ensure_ascii=False), user="root")
    reward_script = (
        "#!/usr/bin/env bash\n"
        f"cd {shlex.quote(str(WORKSPACE))}\n"
        f"OPENCLAW_REWARD_PAYLOAD={shlex.quote(str(payload_path))} bash reward/reward.sh\n"
        "echo $? > /tmp/openclaw_reward.done\n"
    )
    await sb.write_file("/tmp/run_openclaw_reward.sh", reward_script, user="root")
    await sb.exec(
        "rm -rf /tmp/openclaw_reward.spawned && "
        "rm -f /tmp/openclaw_reward.pid /tmp/openclaw_reward.done "
        "/tmp/openclaw_reward.out /tmp/openclaw_reward.err && "
        "chmod +x /tmp/run_openclaw_reward.sh",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )
    launch = (
        "if mkdir /tmp/openclaw_reward.spawned 2>/dev/null; then "
        "setsid /tmp/run_openclaw_reward.sh < /dev/null "
        "> /tmp/openclaw_reward.out 2> /tmp/openclaw_reward.err & "
        "echo $! > /tmp/openclaw_reward.pid; "
        "fi"
    )
    await sb.exec(
        f"bash -c {shlex.quote(launch)}",
        user="root",
        check=True,
        timeout=30,
        idempotent=True,
    )

    reward_exit_code: int | None = None
    stdout = ""
    stderr = ""
    deadline = time.monotonic() + CONFIG.reward_timeout_sec
    while time.monotonic() < deadline:
        ec, out, _ = await sb.exec(
            "test -f /tmp/openclaw_reward.done && cat /tmp/openclaw_reward.done",
            user="root",
            timeout=15,
            idempotent=True,
        )
        if ec == 0 and out.strip():
            try:
                reward_exit_code = int(out.strip())
            except ValueError as exc:
                raise RuntimeError(f"invalid reward exit status for task={task.task_id}: {out!r}") from exc
            stdout = await sb.read_file("/tmp/openclaw_reward.out", user="root")
            stderr = await sb.read_file("/tmp/openclaw_reward.err", user="root")
            break
        await asyncio.sleep(2)
    else:
        await sb.exec(
            "test ! -f /tmp/openclaw_reward.pid || "
            "kill -- -$(cat /tmp/openclaw_reward.pid) 2>/dev/null || true",
            user="root",
            timeout=15,
            idempotent=True,
        )
        raise TimeoutError(f"reward timed out task={task.task_id} after {CONFIG.reward_timeout_sec}s")

    assert reward_exit_code is not None
    if reward_exit_code != 0:
        logger.warning(
            "[openclaw_rl] verifier failed task=%s exit=%d; keeping rollout with reward 0: %s",
            task.task_id,
            reward_exit_code,
            stderr[-1000:],
        )
        return RewardRunResult(score=0.0, verifier_failed=True)
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"reward produced no score for task={task.task_id}")
    try:
        reward = float(lines[-1])
    except ValueError as exc:
        raise RuntimeError(f"reward produced a non-numeric score for task={task.task_id}: {lines[-1]!r}") from exc
    if not math.isfinite(reward):
        raise RuntimeError(f"reward produced a non-finite score for task={task.task_id}: {reward!r}")
    return RewardRunResult(score=max(0.0, min(1.0, reward)), verifier_failed=False)


async def _read_transcript(sb: Sandbox) -> list[dict[str, Any]]:
    _, raw, _ = await sb.exec(
        "find /root/.openclaw/agents/main/sessions -type f -name '*.jsonl' -print 2>/dev/null "
        "| sort | xargs -r cat",
        user="root",
        timeout=30,
        check=True,
        idempotent=True,
    )
    events = []
    lines = raw.splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty_index = nonempty_indices[-1] if nonempty_indices else None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty_index:
                logger.warning("[openclaw_rl] ignoring an incomplete final transcript JSONL line")
                break
            raise RuntimeError("OpenClaw transcript contains malformed JSON") from exc
        if isinstance(item, dict):
            events.append(item)
    return events


async def _upload_tree(sb: Sandbox, source: Path, destination: PurePosixPath) -> None:
    await sb.exec(
        f"mkdir -p {shlex.quote(str(destination))}", user="root", check=True, idempotent=True
    )
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        remote = destination / PurePosixPath(relative.as_posix())
        if path.is_dir():
            await sb.exec(f"mkdir -p {shlex.quote(str(remote))}", user="root", check=True, idempotent=True)
        elif path.is_file():
            await sb.exec(
                f"mkdir -p {shlex.quote(str(remote.parent))}", user="root", check=True, idempotent=True
            )
            await sb.write_file(str(remote), path, user="root")


def _annotate_advisor_call_tokens(
    samples: list[Sample],
    call_outputs: list[list[int]],
    *,
    call_costs_rmb: list[float] | None = None,
) -> None:
    """Mark trained response tokens belonging to assistant cloud-call turns.

    The adapter callback captures the exact sampled output ids for every turn
    that emitted ``cloud_assist``. Trajectory construction may fan out or may
    re-emit a response as loss_mask=0 context, so locate each exact output only
    in a sample where all matched positions remain trainable.
    """
    if call_costs_rmb is None:
        call_costs_rmb = [0.0] * len(call_outputs)
    if len(call_costs_rmb) != len(call_outputs):
        raise ValueError("advisor call costs must align one-to-one with captured call outputs")
    masks: list[list[int]] = [[0] * int(sample.response_length or 0) for sample in samples]
    cost_masks: list[list[float]] = [[0.0] * int(sample.response_length or 0) for sample in samples]
    for output_ids, call_cost in zip(call_outputs, call_costs_rmb, strict=True):
        if not output_ids:
            raise RuntimeError("advisor-call output contains no sampled token ids")
        found = False
        for sample, call_mask, cost_mask in zip(samples, masks, cost_masks, strict=True):
            response_length = int(sample.response_length or 0)
            response_tokens = sample.tokens[-response_length:] if response_length else []
            loss_mask = sample.loss_mask or [1] * response_length
            limit = len(response_tokens) - len(output_ids) + 1
            for start in range(max(0, limit)):
                end = start + len(output_ids)
                if (
                    response_tokens[start:end] == output_ids
                    and all(loss_mask[start:end])
                    and not any(call_mask[start:end])
                ):
                    call_mask[start:end] = [1] * len(output_ids)
                    cost_mask[start:end] = [float(call_cost)] * len(output_ids)
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(
                "could not align advisor-call output to trainable response tokens "
                f"(output_tokens={len(output_ids)})"
            )

    for sample, call_mask, cost_mask in zip(samples, masks, cost_masks, strict=True):
        sample.train_metadata = {
            **(sample.train_metadata or {}),
            "advisor_call_token_mask": call_mask,
            "advisor_call_cost_rmb_mask": cost_mask,
        }


def _session_id(sample: Sample, task_id: str) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"openclaw-{task_id}-{sample.index}-{sample.group_index}"
    return f"openclaw-{task_id}-{secrets.token_hex(8)}"


def _abort_result(
    sample: Sample,
    reason: str,
    *,
    task_id: str,
    reward_role: str,
    agent_exit_code: int | None = None,
    agent_truncated: bool = False,
    agent_process_failure: bool = False,
    verifier_failure: bool = False,
    advisor_stats: dict[str, Any] | None = None,
    agent_turn_stats: dict[str, int] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    episode_timeline: list[dict[str, Any]] | None = None,
) -> list[Sample]:
    reward, advisor_extra = REWARD_MODULE.finalize_rollout(
        0.0,
        advisor_stats,
        penalty_per_call=CONFIG.advisor_penalty_per_call,
        penalty_cap=CONFIG.advisor_penalty_cap,
        role=reward_role,
    )
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "task_id": task_id,
        "agent_exit_code": agent_exit_code,
        "agent_truncated": agent_truncated,
        "agent_process_failure": agent_process_failure,
        "verifier_failure": verifier_failure,
        "transcript_events": len(transcript or []),
        **({"transcript": transcript} if CONFIG.save_transcript and transcript is not None else {}),
        "episode_timeline": episode_timeline or [],
        "agent_turns": int((agent_turn_stats or {}).get("turns", 0) or 0),
        "agent_tool_calls": int((agent_turn_stats or {}).get("tool_calls", 0) or 0),
        "agent_model_requests": int((agent_turn_stats or {}).get("model_requests", 0) or 0),
        "agent_model_responses": int((agent_turn_stats or {}).get("model_responses", 0) or 0),
        "agent_generated_tokens": int((agent_turn_stats or {}).get("generated_tokens", 0) or 0),
        "agent_max_context_tokens": int((agent_turn_stats or {}).get("max_context_tokens", 0) or 0),
        "advisor_followup_model_requests": int(
            (agent_turn_stats or {}).get("advisor_followup_requests", 0) or 0
        ),
        "advisor_followup_model_responses": int(
            (agent_turn_stats or {}).get("advisor_followup_responses", 0) or 0
        ),
        "advisor_force_cloud": CONFIG.advisor_force_cloud,
        "advisor_style": CONFIG.advisor_style,
        **advisor_extra,
        "abort_reason": reason,
    }
    return [sample]
