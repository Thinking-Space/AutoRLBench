"""Frozen reward-agent (judge) client for GUI actor RL.

The judge is a trained GUIRL reward agent served on a dedicated (frozen)
sglang instance with an OpenAI-compatible API. Given an actor trajectory
(task instruction + chronological screenshots + action history), it returns
a verdict: success / failure / uncertain / parse_failure.

Prompt building and output parsing are reused verbatim from
``guirl.judge.judge_prompts`` so the judge sees exactly the distribution it
was trained on.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import aiohttp

from guirl.judge.judge_prompts import (
    JudgeStatus,
    ParsedJudgeOutput,
    parse_judge_output,
)

logger = logging.getLogger(__name__)

# Thinking judge (24k token output) at high concurrency can exceed 600s.
# sock_read keeps the connection alive; total is the hard ceiling.
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=1800, connect=30, sock_read=600)

# Track judge request failures globally (timeout/network/server 5xx) to surface
# in wandb. Reset at module reload (per rollout). Exposed for testing / monitoring.
_judge_request_failed_count = 0


@dataclass
class JudgeResult:
    status: JudgeStatus
    criteria_text: str = ""
    raw_text: str = ""
    attempts: int = 0
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # judge->probe->judge loop bookkeeping (monitoring only)
    num_rounds: int = 1
    used_probe: bool = False
    probe_steps: int = 0


@lru_cache(maxsize=4)
def _get_judge_semaphore(concurrency: int) -> asyncio.Semaphore:
    return asyncio.Semaphore(max(1, concurrency))


class JudgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 1.0,
        max_tokens: int = 24576,
        parse_max_retries: int = 3,
        concurrency: int = 16,
        request_max_retries: int = 3,
        enable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.parse_max_retries = max(1, int(parse_max_retries))
        self.request_max_retries = max(1, int(request_max_retries))
        self.concurrency = max(1, int(concurrency))
        self.enable_thinking = bool(enable_thinking)
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def from_args(cls, args: Any) -> "JudgeClient":
        base_url = getattr(args, "gui_judge_base_url", None) or os.getenv(
            "GUI_JUDGE_BASE_URL", "http://host_0:30100/v1"
        )
        model_name = getattr(args, "gui_judge_model_name", None) or os.getenv("GUI_JUDGE_MODEL_NAME", "guirl-judge")
        return cls(
            base_url=str(base_url),
            model_name=str(model_name),
            # Defaults mirror the judge's TRAINING config (guirl 16x2epoch_2node):
            # temperature=1.0, max_tokens=24576, enable_thinking=true. Keeping
            # these aligned matters — the judge is a frozen thinking model and
            # sees a different distribution if sampled differently than trained.
            temperature=float(getattr(args, "gui_judge_temperature", 1.0) or 1.0),
            max_tokens=int(getattr(args, "gui_judge_max_tokens", 24576) or 24576),
            parse_max_retries=int(getattr(args, "gui_judge_parse_max_retries", 3) or 3),
            concurrency=int(getattr(args, "gui_judge_concurrency", 16) or 16),
            enable_thinking=bool(getattr(args, "gui_judge_enable_thinking", True)),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
        return self._session

    async def _chat(self, messages: list[dict[str, Any]], temperature: float) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            # Match the judge's TRAINING render exactly: guirl applies the chat
            # template with enable_thinking (see slime_custom._apply_judge_chat_template).
            # sglang's OpenAI endpoint forwards chat_template_kwargs to
            # tokenizer.apply_chat_template, so the frozen judge sees the same
            # thinking-mode prompt it was trained on.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        url = f"{self.base_url}/chat/completions"
        session = await self._get_session()
        last_exc: Exception | None = None
        for attempt in range(1, self.request_max_retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        raise RuntimeError(f"judge HTTP {resp.status}: {data}")
                    text = data["choices"][0]["message"]["content"] or ""
                    return text, data.get("usage", {}) or {}
            except Exception as exc:  # network / server hiccups
                last_exc = exc
                if attempt < self.request_max_retries:
                    delay = 2.0 * attempt
                    logger.warning(
                        "judge request attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt,
                        self.request_max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise RuntimeError(f"judge request failed after {self.request_max_retries} attempts: {last_exc}")

    async def chat(self, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        """Raw single chat call, returns response text (no parsing).

        Concurrency is bounded by the shared judge semaphore.
        """
        temp = self.temperature if temperature is None else float(temperature)
        semaphore = _get_judge_semaphore(self.concurrency)
        async with semaphore:
            text, _usage = await self._chat(messages, temperature=temp)
        return text

    async def chat_parseable(
        self,
        messages: list[dict[str, Any]],
        *,
        round_idx: int = 0,
    ) -> tuple[str, JudgeStatus, str, int]:
        """Generate a judge response, retrying parse failures at the SAME temperature.

        Aligned byte-for-byte in behavior with
        ``guirl.slime_custom._generate_parseable_judge_response``: re-generate the
        identical ``messages`` until ``parse_judge_output`` succeeds; after
        ``parse_max_retries`` failures the verdict degrades to ``parse_failure``
        (never raises — the caller drops the sample). The retry KEEPS
        ``self.temperature`` (training uses temperature=1.0 on every attempt); it
        does NOT raise temperature.

        Returns ``(response_text, status, criteria_text, attempts)``. Concurrency
        is bounded by the shared judge semaphore.
        """
        semaphore = _get_judge_semaphore(self.concurrency)
        last_error: Exception | None = None
        last_response = ""
        async with semaphore:
            for attempt in range(1, self.parse_max_retries + 1):
                try:
                    response_text, _usage = await self._chat(messages, temperature=self.temperature)
                except Exception as exc:
                    # Network/server failure is not a parse failure, but there is
                    # nothing usable to train on → degrade to parse_failure.
                    global _judge_request_failed_count
                    _judge_request_failed_count += 1
                    logger.warning(
                        "judge request failed (round %d, attempt %d/%d): %s — sample will be dropped (total_failed=%d)",
                        round_idx,
                        attempt,
                        self.parse_max_retries,
                        exc,
                        _judge_request_failed_count,
                        extra={"judge_request_failed": 1},
                    )
                    return "", "parse_failure", "", attempt

                last_response = response_text
                try:
                    parsed: ParsedJudgeOutput = parse_judge_output(response_text)
                except ValueError as exc:
                    last_error = exc
                    logger.warning(
                        "judge parse failed (round %d, attempt %d/%d): %s",
                        round_idx,
                        attempt,
                        self.parse_max_retries,
                        exc,
                    )
                    continue
                return response_text, parsed.status, parsed.criteria_text, attempt

        logger.warning("judge parse exhausted retries (round %d): %s", round_idx, last_error)
        return last_response, "parse_failure", "", self.parse_max_retries

    async def close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
