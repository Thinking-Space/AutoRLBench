"""Logic layer of the session server: ``SessionCore``.

HTTP-agnostic: the FastAPI adapter (``sessions.py`` + ``server.py``) turns each request into primitives and calls these methods. Owns one ``SessionRegistry`` (per-session TITO/trajectory state) and one proxy ``backend``.

- ``chat_completions`` strips the R3 replay payloads (``routed_experts`` / ``indexer_topk``) from the client reply copy-on-write; the ``SessionRecord`` keeps the full response for the training path (``GET /sessions/{id}``).
- ``chat_completions`` holds the per-session lock for prep and state update but not across the proxy call; ``closing`` re-checks and the ``num_assistant`` check gate concurrent DELETE/chat.
- ``stream: true`` is served as fake streaming: the backend call stays non-streaming (TITO needs the complete message + meta_info) and the full response is re-rendered as a single SSE chunk plus ``data: [DONE]``. Errors all happen before the SSE body is built, so they keep their real status codes as JSON.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from starlette.responses import Response

from miles.rollout.engine_client.client import EngineClientDrainingError
from miles.rollout.engine_client.client_sdk import render_chat_completions_response
from miles.rollout.session.errors import (
    EngineDrainingError,
    HistoryPrefixMismatchError,
    MessageValidationError,
    SessionConfigurationError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.provider import ProviderConfigurationError, SessionProfile
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils.template import normalize_tool_arguments

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"

# Hop-by-hop / length-framing headers dropped from the upstream response so the
# transport layer recomputes them from the body we actually send. "server" and
# "date" are dropped because our own ASGI server always emits them, so echoing
# upstream's copy puts two of each on the wire; aiohttp's parser rejects that
# outright with "Duplicate 'Server' header found" instead of reading the body.
_DROP_RESPONSE_HEADERS = ("content-length", "transfer-encoding", "content-encoding", "server", "date")


@dataclass
class ProxyRequest:
    """Primitive carrier for the proxy backend (replaces fastapi.Request)."""

    method: str
    query: str = ""


def _render_json(payload) -> bytes:
    """Encode like Starlette's JSONResponse (compact, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


_CLIENT_STRIPPED_META_KEYS = ("routed_experts", "indexer_topk")


def _strip_replay_payloads(response: dict, keys: tuple[str, ...] = _CLIENT_STRIPPED_META_KEYS) -> dict:
    stripped_choices = []
    for choice in response.get("choices", []):
        meta = choice.get("meta_info")
        if isinstance(meta, dict) and any(k in meta for k in keys):
            meta = {k: v for k, v in meta.items() if k not in keys}
            choice = {**choice, "meta_info": meta}
        stripped_choices.append(choice)
    return {**response, "choices": stripped_choices}


def _client_stripped_keys(session, response: dict) -> tuple[str, ...]:
    """Let the provider consume the upstream extras and choose what to strip.

    A provider without ``consume_meta`` keeps the module-level default list, so
    the client copy is byte-identical to what the core produced before.
    """
    consume = getattr(session, "consume_meta", None)
    if not callable(consume):
        return _CLIENT_STRIPPED_META_KEYS
    keys = consume(response)
    if keys is None:
        return _CLIENT_STRIPPED_META_KEYS
    return tuple(keys)


def _response_to_stream_chunk(response: dict) -> dict:
    """Synthesize the single ``chat.completion.chunk`` for a fake stream.

    Adapted from NVIDIA-NeMo/ProRL-Agent-Server (``gateway/server.py::_response_to_stream_chunk``)
    and THUDM/slime (``agent/adapters/openai.py::_render_stream``).

    One big delta is protocol-legal (streaming deltas concatenate). All
    tool_calls ride in this one chunk with their ``index`` set: some clients
    mis-assemble arguments fragmented across chunks. The chunk carries no
    ``meta_info``; the training path reads ``GET /sessions/{id}`` instead.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    delta = {"role": message.get("role", "assistant"), "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{**tool_call, "index": i} for i, tool_call in enumerate(message["tool_calls"])]
    chunk = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}],
    }
    if response.get("usage") is not None:
        chunk["usage"] = response["usage"]
    return chunk


def _chat_client_response(
    result: dict,
    response: dict,
    client_stream: bool,
    stripped_keys: tuple[str, ...] = _CLIENT_STRIPPED_META_KEYS,
) -> Response:
    if client_stream:
        sse = b"data: " + _render_json(_response_to_stream_chunk(response)) + b"\n\ndata: [DONE]\n\n"
        # Fresh headers: upstream's headers describe its JSON body, not this SSE body.
        # X-Accel-Buffering keeps reverse proxies from buffering the stream.
        return Response(
            content=sse,
            status_code=result["status_code"],
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            media_type="text/event-stream",
        )
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return Response(
        content=_render_json(_strip_replay_payloads(response, stripped_keys)),
        status_code=result["status_code"],
        headers=headers,
        media_type=JSON_MEDIA_TYPE,
    )


def proxy_result_to_response(result: dict) -> Response:
    """Build the client response from a proxy result.

    Mirrors the previous ``SessionServer.build_proxy_response``: re-emit JSON
    bodies as compact JSON (application/json), pass non-JSON bodies through
    unchanged, and drop wire-level framing headers from upstream.
    """
    content = result["response_body"]
    status_code = result["status_code"]
    headers = {k: v for k, v in result["headers"].items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    content_type = headers.get("content-type", "")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Match the old Response(media_type=content_type): pass it through verbatim
        # (incl. "" when upstream sent no content-type) so the wire bytes are identical.
        return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)
    return Response(content=_render_json(data), status_code=status_code, headers=headers, media_type=JSON_MEDIA_TYPE)


def _build_tool_call_parser(tito_tokenizer, tools: list[dict] | None) -> FunctionCallParser | None:
    if not tools or not tito_tokenizer.tool_call_parser:
        return None
    return FunctionCallParser(
        tools=TypeAdapter(list[Tool]).validate_python(tools),
        tool_call_parser=tito_tokenizer.tool_call_parser,
    )


class SessionCore:
    """HTTP session operations over one ``SessionRegistry``."""

    def __init__(self, backend, registry: SessionRegistry, args, session_server_instance_id=None, *, engine_client=None):
        self.backend = backend
        self.registry = registry
        self.args = args
        self.instance_id = session_server_instance_id
        # Set when the partial-rollout gateway is available; chat_completions
        # only routes through it when --rollout-partial-mode=token, so None (or
        # the mode being off) simply keeps the plain proxy path.
        self.engine_client = engine_client
        # Weight-update coordination. The gate starts open, so a server that
        # never calls quiesce behaves exactly as it did before this existed.
        self._accepting_generations = True
        self._inflight_generations = 0
        self._generation_gate = asyncio.Condition()
        # Two-phase export bookkeeping: a session enters _exports_offered when a
        # caller reads its summary, and _exports_acknowledged when it confirms
        # receipt. Used for one warning on DELETE, nothing else.
        self._exports_offered: set[str] = set()
        self._exports_acknowledged: set[str] = set()

    # --- weight-update coordination ---------------------------------------

    def drain_status(self) -> dict:
        return {
            "accepting_generations": self._accepting_generations,
            "inflight_generations": self._inflight_generations,
            "drained": not self._accepting_generations and self._inflight_generations == 0,
        }

    async def quiesce(self, timeout: float = 300.0) -> dict:
        """Stop admitting new turns and wait for in-flight ones to land."""
        async with self._generation_gate:
            self._accepting_generations = False
            self._generation_gate.notify_all()
            try:
                await asyncio.wait_for(
                    self._generation_gate.wait_for(lambda: self._inflight_generations == 0),
                    timeout=timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):
                # Report honestly rather than hanging: the caller decides whether
                # to swap weights anyway or to resume and retry later.
                status = self.drain_status()
                status.update({"drained": False, "timed_out": True})
                return status
            status = self.drain_status()
            status["timed_out"] = False
            return status

    async def resume(self) -> dict:
        async with self._generation_gate:
            self._accepting_generations = True
            self._generation_gate.notify_all()
            return self.drain_status()

    async def health(self) -> Response:
        body = {"status": "ok"}
        if self.instance_id is not None:
            body["session_server_instance_id"] = self.instance_id
        return Response(content=_render_json(body), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def create_session(self, profile: SessionProfile | None = None) -> Response:
        try:
            session_id = self.registry.create_session(profile)
        except ProviderConfigurationError as exc:
            raise SessionConfigurationError(str(exc)) from exc
        selected = profile or SessionProfile()
        payload = {"session_id": session_id}
        if selected.name != "linear" or (profile is not None and selected.metadata):
            payload["profile"] = selected.name
        # A provider may append keys to the create response — in practice the
        # per-rollout bearer token, which is generated here and has no other way
        # of reaching the agent. Providers without the hook add nothing, so the
        # two response shapes above stay byte-identical.
        provider = self.registry.get_session(session_id)
        create_result = getattr(provider, "create_result", None)
        if callable(create_result):
            extra = create_result() or {}
            conflicts = set(extra) & set(payload)
            if conflicts:
                raise SessionConfigurationError(f"provider create_result may not override {sorted(conflicts)}")
            payload.update(extra)
        return Response(
            content=_render_json(payload),
            status_code=200,
            media_type=JSON_MEDIA_TYPE,
        )

    async def capabilities(self, profile_name: str | None = None) -> Response:
        try:
            payload = self.registry.provider_registry.capabilities(profile_name)
        except ProviderConfigurationError as exc:
            raise SessionConfigurationError(str(exc)) from exc
        return Response(content=_render_json(payload), status_code=200, media_type=JSON_MEDIA_TYPE)

    async def get_session(self, session_id: str, *, view: str | None = None) -> Response:
        if view not in (None, "full", "summary"):
            raise MessageValidationError(f"unknown view: {view!r} (expected 'full' or 'summary')")
        session = self.registry.get_session(session_id)
        if view == "summary":
            self._exports_offered.add(session_id)
            return Response(
                content=_render_json(self._session_summary(session_id, session)),
                status_code=200,
                media_type=JSON_MEDIA_TYPE,
            )
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        session_metrics = self.registry.session_metrics_summary(session_id)
        if session_metrics is not None:
            metadata["session_metrics"] = session_metrics
        # Same probe the summary view uses, for the same reason: a provider knows
        # things about its own trajectory that the core cannot derive from the
        # records -- a branching provider's per-branch training segments, for
        # instance, which are not recoverable from a flat record list. Absent or
        # empty leaves the payload exactly as it was.
        export = getattr(session, "export", None)
        if callable(export):
            metadata.update(export("full") or {})
        payload = GetSessionResponse(session_id=session_id, records=session.records, metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    def _session_summary(self, session_id: str, session) -> dict:
        """Counts only — no records, no token IDs, no logprobs, no replay payloads.

        The point of the summary view is to let a caller decide whether a
        trajectory is worth fetching without moving the bulky parts.
        """
        records = session.records
        prompt_tokens = 0
        completion_tokens = 0
        for record in records:
            usage = record.response.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
        summary = {
            "session_id": session_id,
            "turns": len(records),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "trainable_tokens": len(session.token_ids),
            },
        }
        sequence_ids = [record.sequence_id for record in records if record.sequence_id is not None]
        if sequence_ids:
            counts: dict[str, int] = {}
            for sequence_id in sequence_ids:
                counts[sequence_id] = counts.get(sequence_id, 0) + 1
            summary["sequences"] = [
                {"sequence_id": sequence_id, "turns": turns} for sequence_id, turns in counts.items()
            ]
        export = getattr(session, "export", None)
        if callable(export):
            summary.update(export("summary") or {})
        return summary

    async def acknowledge_export(self, session_id: str, body: dict) -> Response:
        """Confirm that a caller received the trajectory it was handed.

        Catches "the data was lost in transit and nobody noticed": the caller
        reports back what it counted, and a disagreement is a 409.
        """
        session = self.registry.get_session(session_id)
        acknowledge = getattr(session, "acknowledge_export", None)
        if callable(acknowledge):
            acknowledge(body)
        else:
            expected = len(session.records)
            reported = body.get("turns")
            if reported is not None and int(reported) != expected:
                raise HistoryPrefixMismatchError(
                    f"export acknowledgement mismatch for session {session_id}: "
                    f"reported turns={reported}, server recorded {expected}"
                )
        self._exports_acknowledged.add(session_id)
        return Response(
            content=_render_json({"session_id": session_id, "acknowledged": True}),
            status_code=200,
            media_type=JSON_MEDIA_TYPE,
        )

    async def delete_session(self, session_id: str) -> Response:
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        if session_id in self._exports_offered and session_id not in self._exports_acknowledged:
            # Somebody asked for a summary and never came back for — or never
            # confirmed — the full trajectory. Worth a line in the log; not worth
            # changing the response, which existing callers depend on.
            logger.warning("Session %s is being deleted without acknowledging its export", session_id)
        self._exports_offered.discard(session_id)
        self._exports_acknowledged.discard(session_id)
        session.closing = True
        # Acquire the lock so an in-flight chat finishes before we drop the session.
        await session.lock.acquire()
        try:
            close_result = self.registry.remove_session(session_id)
        finally:
            session.lock.release()
        if close_result.metadata:
            return Response(
                content=_render_json({"status": close_result.status, "metadata": dict(close_result.metadata)}),
                status_code=200,
                media_type=JSON_MEDIA_TYPE,
            )
        return Response(status_code=204)

    @asynccontextmanager
    async def _admit_turn(self):
        """Admission gate plus in-flight accounting for one generation.

        Every exit path — including exceptions — decrements exactly once.
        Leaking a count would make ``quiesce`` wait forever, and it would only
        show up during a weight update.
        """
        async with self._generation_gate:
            await self._generation_gate.wait_for(lambda: self._accepting_generations)
            self._inflight_generations += 1
        try:
            yield
        finally:
            async with self._generation_gate:
                self._inflight_generations -= 1
                self._generation_gate.notify_all()

    async def chat_completions(
        self,
        session_id: str,
        *,
        method: str,
        query: str,
        headers: dict,
        body: bytes,
        record_path: str = "/v1/chat/completions",
    ) -> Response:
        """Run one recorded chat turn.

        ``record_path`` labels the turn in the session's records. It exists for
        route extensions (see ``--session-route-paths``): a request that arrived
        as OpenAI Responses or Anthropic Messages is translated to chat before it
        gets here, and the record should say which protocol it came in on rather
        than claiming every turn was chat.
        """
        async with self._admit_turn():
            return await self._run_turn(
                session_id,
                method=method,
                query=query,
                headers=headers,
                body=body,
                record_path=record_path,
            )

    async def _run_turn(
        self,
        session_id: str,
        *,
        method: str,
        query: str,
        headers: dict,
        body: bytes,
        record_path: str = "/v1/chat/completions",
    ) -> Response:
        """Proxy a chat completion through the backend with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (NO lock) → validate response → update trajectory checkpoint and
        append record (lock held briefly). The lock is NOT held during the slow
        proxy call so DELETE/other ops are not blocked if the agent disconnects.
        """
        request_timestamp = time.time()
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            try:
                request_body = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                raise MessageValidationError(f"invalid JSON body: {e}") from e

            # Fake streaming: the backend must stay non-streaming (TITO needs the
            # complete message + meta_info, and sglang rejects return_meta_info
            # with stream=true), so pop the client's intent here and honor it
            # when rendering the client response.
            client_stream = bool(request_body.pop("stream", False))
            request_body.pop("stream_options", None)

            # TITO token tracking needs Miles-owned input_ids plus SGLang output
            # metadata: logprobs=True populates meta_info.output_token_logprobs and
            # return_meta_info wraps it in choice.meta_info. Hardcoded (not
            # setdefault) so agent-side overrides cannot break token accumulation.
            request_body["logprobs"] = True
            request_body["return_meta_info"] = True
            if getattr(self.args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            if getattr(self.args, "use_rollout_indexer_replay", False):
                request_body["return_indexer_topk"] = True
            # A provider may append or override upstream fields here — this is
            # where an incremental replay cursor goes. Placed after the switches
            # above so it can override them, and before the invariants below so
            # it cannot break token accumulation.
            extra_fields = getattr(session, "extra_request_fields", None)
            if callable(extra_fields):
                request_body.update(extra_fields() or {})
            # Must be False so stop-token text is trimmed from assistant content;
            # token IDs still come from logprobs below.
            request_body["no_stop_trim"] = False
            # FIXME(session): Only nested `chat_template_kwargs` reach the local renderer;
            # top-level `reasoning` and `reasoning_effort` are not mapped to template kwargs.
            request_ctk = request_body.get("chat_template_kwargs")
            if request_ctk is not None and not isinstance(request_ctk, dict):
                raise MessageValidationError("chat_template_kwargs must be an object")
            tito_tokenizer = self.registry.tito_tokenizer
            if request_ctk:
                try:
                    tito_tokenizer = tito_tokenizer.clone_with_chat_template_kwargs(request_ctk)
                except ValueError as e:
                    raise MessageValidationError(str(e)) from e
            if tito_tokenizer.chat_template_kwargs:
                request_body["chat_template_kwargs"] = dict(tito_tokenizer.chat_template_kwargs)
            else:
                request_body.pop("chat_template_kwargs", None)

            # The same messages feed two consumers that disagree about how
            # tool-call arguments are spelled, so normalize twice rather than
            # letting one of them lose:
            #
            #   prepare_pretokenized -> the HF-Jinja chat template, which indexes
            #       arguments as an object (MiniCPM5 does args_dict.items()).
            #   proxy_body -> SGLang, which sits behind a Rust router that
            #       deserializes function.arguments into a String and refuses a
            #       map ("messages[N]: invalid type: map, expected a string").
            #
            # Sending the dict upstream 400'd every agent turn whose history
            # carried a tool call -- i.e. the second turn onward, since the first
            # request has none. codex aborted 8/8 rollout groups on this with no
            # training step (965951/966122).
            request_messages = request_body.get("messages", [])
            try:
                template_messages = normalize_tool_arguments(request_messages, "dict")
                request_body["messages"] = normalize_tool_arguments(request_messages, "json")
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                raise MessageValidationError(f"invalid tool-call arguments: {e}") from e
            # Snapshot before prepare_pretokenized so the turn can be classified as
            # continued / branched against the prefix it was actually built on.
            previous_token_ids = list(session.token_ids)
            prompt_token_ids = session.prepare_pretokenized(
                template_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=tito_tokenizer,
            )
            request_body["input_ids"] = prompt_token_ids
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            tool_call_parser = _build_tool_call_parser(tito_tokenizer, request_body.get("tools"))
            proxy_body = json.dumps(request_body).encode()
            expected_generation = session.generation
        # --- lock released ---

        # --- Phase 2: proxy to backend (NO lock held) ---
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        use_gateway = self.engine_client is not None and getattr(self.args, "rollout_partial_mode", "off") == "token"
        if use_gateway:
            try:
                turn_record = await self.engine_client.call(
                    session_id=session_id,
                    branch_id=session_id,
                    turn_index=session.num_assistant,
                    path="chat_completions",
                    body=request_body,
                    headers=headers,
                )
            except EngineClientDrainingError as e:
                raise EngineDrainingError(str(e)) from e
            response = render_chat_completions_response(
                turn_record, tokenizer=self.registry.tokenizer, tool_call_parser=tool_call_parser
            )
            result = {
                "request_body": proxy_body,
                "response_body": _render_json(response),
                "status_code": 200,
                "headers": {"content-type": JSON_MEDIA_TYPE},
            }
        else:
            result = await self.backend.do_proxy(
                ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers
            )

            # Non-200 (e.g. 400 context too long) passes through unrecorded so the
            # agent can retry or handle the error.
            if result["status_code"] != 200:
                return proxy_result_to_response(result)

            response = json.loads(result["response_body"])
        choice = response.get("choices", [{}])[0]

        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError(
                "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
            )
        assistant_message = choice.get("message") or {}
        if assistant_message.get("content") is None:
            raise UpstreamResponseError(
                "assistant message content is None, when tool call parser failed SGLang should still return "
                "an empty content rather than None. Please check your modified SGLang version."
            )

        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info["completion_tokens"]

        actual_output_logprobs_len = len(output_token_logprobs)
        if actual_output_logprobs_len != completion_tokens:
            raise UpstreamResponseError(
                "invalid chat completion response: "
                f"len(output_token_logprobs)={actual_output_logprobs_len} "
                f"!= completion_tokens={completion_tokens}. "
                f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
            )

        completion_token_ids = [t[1] for t in output_token_logprobs]

        # The provider takes what it wants out of the upstream extras and says
        # what to hide from the client. Without the hook this is the default list.
        stripped_keys = _client_stripped_keys(session, response)

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            if session.closing:
                logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                return _chat_client_response(result, response, client_stream, stripped_keys)

            if session.generation != expected_generation:
                logger.warning(
                    f"Session {session_id} state changed during proxy "
                    f"(expected generation={expected_generation}, "
                    f"got {session.generation}), skipping state update"
                )
                return _chat_client_response(result, response, client_stream, stripped_keys)

            update_kwargs = {
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "max_trim_tokens": self.registry.tito_tokenizer.max_trim_tokens,
            }
            if getattr(session, "accepts_completion_metadata", False):
                update_kwargs["completion_metadata"] = {
                    **meta_info,
                    "finish_reason": choice.get("finish_reason"),
                }
            session.update_pretokenized_state(
                template_messages,
                assistant_message,
                **update_kwargs,
            )

            record = SessionRecord(
                timestamp=time.time(),
                request_timestamp=request_timestamp,
                method=method,
                path=record_path,
                status_code=result["status_code"],
                request=request_body,
                response=response,
                sequence_id=getattr(session, "active_sequence_id", None),
            )
            session.append_record(record)
            self.registry.record_turn_metrics(
                session_id,
                previous_token_ids=previous_token_ids,
                prompt_token_ids=prompt_token_ids,
                assistant_message=assistant_message,
            )
        # --- lock released ---

        return _chat_client_response(result, response, client_stream, stripped_keys)

    async def proxy(
        self, session_id: str, path: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), path, body=body, headers=headers
        )
        return proxy_result_to_response(result)

    # --- alternative wire protocols ---------------------------------------
    # These reuse the chat path wholesale: same locking, same token accounting,
    # same trajectory recording. Only the request/response shape differs.

    def count_tokens(
        self,
        session_id: str,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> int:
        """Token count for a prompt or an *already normalized* message list.

        The normalization this deliberately does not do -- flattening structured
        content, canonicalizing tool calls -- is protocol-shaped, and protocol
        translation belongs to the route extension that received the request (see
        ``--session-route-paths``). What is left here is the part only the server
        can do: it owns the tokenizer and the chat template.

        Exactly one of ``messages`` / ``prompt``; passing both or neither is a
        client error, not a server one.
        """
        if (messages is None) == (prompt is None):
            raise MessageValidationError("count_tokens requires exactly one of 'messages' or 'prompt'")
        self.registry.get_session(session_id)
        if prompt is not None:
            return len(self.registry.tokenizer.encode(str(prompt), add_special_tokens=False))
        return len(
            self.registry.tito_tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
        )

    async def models(self, session_id: str) -> Response:
        self.registry.get_session(session_id)
        model_id = getattr(self.args, "hf_checkpoint", None) or "miles"
        payload = {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "miles-session"}],
        }
        return Response(content=_render_json(payload), status_code=200, media_type=JSON_MEDIA_TYPE)

    def session_token(self, session_id: str) -> str | None:
        """The per-rollout credential a provider minted, if it minted one.

        Deliberately silent about unknown sessions. This runs before the real
        handler, and the handlers disagree about what an unknown session means:
        ``chat_completions`` raises 404, the catch-all proxy forwards the
        request upstream without looking. Raising here would impose the former
        on both and change what the proxy route returns.
        """
        session = self.registry.sessions.get(session_id)
        return getattr(session, "session_token", None) if session is not None else None
