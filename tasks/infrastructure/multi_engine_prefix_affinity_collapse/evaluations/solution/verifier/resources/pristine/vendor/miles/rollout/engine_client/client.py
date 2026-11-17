"""EngineClient: token-level partial-rollout / fully-async client core.

Owns the abort -> resend loop for a single logical generation call, hiding
segment boundaries from the caller: ``call()`` always returns one complete
``TurnRecord`` (or raises); it never returns a half-finished result. This is
the sole component that gets to see individual abort segments -- everything
downstream (miles training side, existing response parsers) only ever sees
a turn that looks complete.

Also owns EngineClient's own HTTP surface: the control router
(``/control/drain``, ``/control/stats``), the stateless white-box generate
router (``create_engine_client_router``, mounted by
``session/sessions.py``), and the optional Prometheus push for its stats
(``push_engine_client_stats``) -- all three only ever act on/report this
module's own ``EngineClient``, so they live together rather than split
across separate files.
"""

import copy
import logging
import time

from fastapi import APIRouter, Request

from miles.rollout.engine_client.client_sdk import render_generate_response
from miles.rollout.engine_client.meta_info import parse_meta_info
from miles.rollout.engine_client.types import GenerationSegment, TurnRecord
from miles.utils.http_utils import filter_forwarded_headers
from miles.utils.http_utils import post as _default_post

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "rollout/gateway"


def _extract_max_new_tokens(body: dict, *, path: str) -> int | None:
    if path == "generate":
        return (body.get("sampling_params") or {}).get("max_new_tokens")
    return body.get("max_completion_tokens", body.get("max_tokens"))


def _extract_max_tokens_field(body: dict) -> str:
    """Which chat-completions token-limit field the caller used.

    EngineClient doesn't control the caller's field convention (both
    ``max_tokens`` and the non-deprecated ``max_completion_tokens`` are
    accepted), so every resend must write back to the same field the caller
    originally sent -- never a hardcoded one -- to stay wire-compatible with
    ``do_proxy``.
    """
    return "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"


class EngineClientDrainingError(RuntimeError):
    def __init__(self, *, session_id: str, turn_index: int):
        super().__init__(f"EngineClient is draining; refusing to resume session={session_id} turn={turn_index}")
        self.session_id = session_id
        self.turn_index = turn_index


class EngineClient:
    """Talks to the real inference-engine backend (the same URL
    single_turn.py/multi_turn.py would use directly in local mode) and
    performs the abort/resume loop when ``--rollout-partial-mode token`` is
    set. With the default ``off``, it behaves as a thin single-shot proxy
    (no resume), so enabling the remote transport alone does not change
    generation semantics.
    """

    def __init__(self, backend_url: str, *, args, timeout: float = 600.0, post_fn=_default_post):
        self._backend_url = backend_url.rstrip("/")
        self._args = args
        self._timeout = timeout
        self._post = post_fn
        self._resume_enabled = getattr(args, "rollout_partial_mode", "off") == "token"
        self._draining = False
        self._calls_total = 0
        self._segments_total = 0
        self._resumed_total = 0

    def begin_drain(self) -> None:
        """Stop resuming aborted calls; future aborts raise instead."""
        self._draining = True

    def stats(self) -> dict:
        return {
            "calls_total": self._calls_total,
            "segments_total": self._segments_total,
            "resumed_total": self._resumed_total,
            "avg_segments_per_turn": (self._segments_total / self._calls_total) if self._calls_total else 0.0,
        }

    async def call(
        self, *, session_id: str, branch_id: str, turn_index: int, path: str, body: dict, headers: dict | None = None
    ) -> TurnRecord:
        assert path in ("generate", "chat_completions"), f"EngineClient.call: unknown path {path!r}"
        self._calls_total += 1

        prompt_token_ids = tuple(body["input_ids"])
        max_new_tokens_remaining = _extract_max_new_tokens(body, path=path)
        max_tokens_field = _extract_max_tokens_field(body) if path == "chat_completions" else None
        current_input_ids = list(prompt_token_ids)
        segments: list[GenerationSegment] = []
        headers = filter_forwarded_headers(headers)

        while True:
            segment = await self._call_once(
                body=body,
                path=path,
                input_ids=current_input_ids,
                max_new_tokens=max_new_tokens_remaining,
                max_tokens_field=max_tokens_field,
                prefix_len=len(current_input_ids),
                headers=headers,
            )
            segments.append(segment)
            self._segments_total += 1

            if segment.finish_reason != "abort" or not self._resume_enabled:
                break
            if self._draining:
                raise EngineClientDrainingError(session_id=session_id, turn_index=turn_index)

            self._resumed_total += 1
            current_input_ids = current_input_ids + list(segment.token_ids)
            if max_new_tokens_remaining is not None:
                max_new_tokens_remaining = max(0, max_new_tokens_remaining - len(segment.token_ids))
                if max_new_tokens_remaining == 0:
                    break

        return TurnRecord(
            turn_index=turn_index,
            prompt_token_ids=prompt_token_ids,
            segments=tuple(segments),
            engine_id=branch_id or session_id,
        )

    async def _call_once(
        self,
        *,
        body: dict,
        path: str,
        input_ids: list[int],
        max_new_tokens: int | None,
        max_tokens_field: str | None,
        prefix_len: int,
        headers: dict | None,
    ) -> GenerationSegment:
        request_body = copy.deepcopy(body)
        request_body["input_ids"] = input_ids
        if max_new_tokens is not None:
            if path == "generate":
                request_body.setdefault("sampling_params", {})["max_new_tokens"] = max_new_tokens
            else:
                other_field = "max_tokens" if max_tokens_field == "max_completion_tokens" else "max_completion_tokens"
                request_body[max_tokens_field] = max_new_tokens
                request_body.pop(other_field, None)

        url = f"{self._backend_url}/generate" if path == "generate" else f"{self._backend_url}/v1/chat/completions"
        start_time = time.time()
        output = await self._post(url, request_body, headers=headers)
        elapsed_seconds = time.time() - start_time
        return parse_meta_info(output, args=self._args, path=path, prefix_len=prefix_len, elapsed_seconds=elapsed_seconds)


def create_control_router(engine_client: EngineClient) -> APIRouter:
    """``POST /control/drain`` + ``GET /control/stats`` for the EngineClient process."""
    router = APIRouter()

    @router.post("/control/drain")
    async def drain():
        engine_client.begin_drain()
        return {"status": "draining"}

    @router.get("/control/stats")
    async def stats():
        snapshot = engine_client.stats()
        push_engine_client_stats(engine_client._args, snapshot)
        return snapshot

    return router


def create_engine_client_router(backend, args, *, tokenizer) -> APIRouter:
    """Stateless white-box HTTP surface for EngineClient.

    Bypasses ``SessionRegistry``/``LinearTrajectory`` entirely (those belong
    to the black-box TITO state machine); each call renders a complete
    synthetic sglang ``/generate`` response so existing parsers don't need to
    change.

    ``backend`` is the ``SessionServer`` instance (same object passed to
    ``SessionCore``); its shared ``engine_client`` is reused here so the
    white-box and black-box (``SessionCore.chat_completions``) paths talk to
    the same in-process ``EngineClient``, per-process ledger and stats."""
    engine_client = backend.engine_client

    router = APIRouter()
    router.include_router(create_control_router(engine_client))

    @router.post("/sessions/{session_id}/generate")
    async def generate(session_id: str, request: Request):
        body = await request.json()
        turn_record = await engine_client.call(
            session_id=session_id,
            branch_id=session_id,
            turn_index=0,
            path="generate",
            body=body,
            headers=None,
        )
        return render_generate_response(turn_record, tokenizer=tokenizer)

    return router


def push_engine_client_stats(args, engine_client_stats: dict) -> None:
    """Optional Grafana/Prometheus surface for EngineClient stats.

    Zero-touch on the training loop: this never calls
    ``miles.utils.tracking_utils.init_tracking`` itself -- it only piggybacks
    on a Prometheus collector the training driver may have already
    initialized (via ``--use-prometheus``). If prometheus isn't initialized
    (or isn't enabled), pushing is a silent no-op.
    """
    if not getattr(args, "use_prometheus", False):
        return
    try:
        from miles.utils.tracking_utils.prometheus_utils import get_prometheus
    except ImportError:
        return

    prom = get_prometheus()
    if prom is None:
        return

    metrics = {f"{_METRIC_PREFIX}/{key}": value for key, value in engine_client_stats.items()}
    try:
        prom.update.remote(metrics)
    except Exception:
        logger.warning("[engine_client] failed to push metrics to prometheus", exc_info=True)
