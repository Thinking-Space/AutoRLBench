"""miles-side helpers for the token-level partial-rollout EngineClient.

``render_generate_response`` turns a completed ``TurnRecord`` back into a
dict shaped exactly like a native sglang ``/generate`` response, so
unmodified parsers (e.g.
``generate_utils.generate_endpoint_utils.update_sample_from_response``)
consume it without knowing EngineClient was involved.

``render_chat_completions_response`` does the same for the black-box TITO
path (``SessionCore.chat_completions``): it re-decodes the full output text
locally (rather than trusting any one segment's own text, which may be
incomplete at an abort boundary) and, when a ``tool_call_parser`` is given,
re-parses tool calls from that complete text -- the same client-side
re-parse pattern ``generate_hub/multi_turn.py`` already uses
(``tool_call_parser.parse_non_stream(output["text"])``).

``validate_continuity`` is a black-box (TITO) helper: unit-tested now but not
called anywhere yet -- wiring it into ``SessionCore`` is deferred to the
integration phase, once the branch-trajectory-manager owner settles
``SessionRecord``'s eventual ``turn_index`` field (see engine_client/README
notes in the design doc, seam 3). It works off today's ``SessionRecord``
shape (list position as the turn-index proxy) so it doesn't block on that.
"""

import numpy as np

from miles.rollout.engine_client.replay import ReplayLedger, encode_replay_buffer
from miles.rollout.engine_client.types import GenerationSegment, TurnRecord

# Additive integer spec counters only. Averages (``spec_accept_rate`` /
# ``spec_accept_length``) and histograms are not summed across abort
# segments; downstream ``Sample.SpecInfo`` recomputes the rates.
_SPEC_COUNTER_KEYS = (
    "spec_accept_token_num",
    "spec_draft_token_num",
    "spec_verify_ct",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_accepted_drafts",
    "spec_proposed_drafts",
)


def render_generate_response(turn_record: TurnRecord, *, tokenizer) -> dict:
    token_ids = turn_record.output_token_ids
    logprobs = turn_record.output_logprobs
    meta_info = {
        "output_token_logprobs": list(zip(logprobs, token_ids, strict=True)),
        "completion_tokens": len(token_ids),
        "finish_reason": {"type": turn_record.finish_reason},
    }
    _attach_engine_client_observability(meta_info, turn_record)
    _attach_spec_counters(meta_info, turn_record)

    total_edges = len(turn_record.prompt_token_ids) + len(token_ids) - 1
    _attach_replay(meta_info, turn_record, field="routed_experts", total_edges=total_edges)
    _attach_replay(meta_info, turn_record, field="indexer_topk", total_edges=total_edges)

    return {"text": tokenizer.decode(token_ids, skip_special_tokens=False), "meta_info": meta_info}


def render_chat_completions_response(turn_record: TurnRecord, *, tokenizer, tool_call_parser=None) -> dict:
    """Render a ``TurnRecord`` as an OpenAI-shaped ``/v1/chat/completions``
    response, compatible with what ``SessionCore.chat_completions`` expects
    from the real backend (``choice.message.content`` /
    ``choice.meta_info.output_token_logprobs`` / ``choice.meta_info.completion_tokens``).

    When the turn has exactly one segment (no abort ever split it), the raw
    engine response fields are preserved, with routed-experts base64 restored
    from compact storage, so the result stays byte-identical to ``do_proxy``.
    Text reconstruction below is only needed for a
    genuine multi-segment turn, where no single wire response represents the
    spliced-together result (see ``EngineClient``'s module docstring for why
    per-segment tool-call parsing can't be trusted across an abort boundary).

    Known limitation (left visible rather than papered over): ``core.py``
    requests ``no_stop_trim=False``, which trims the stop sequence's text
    from the backend's own assistant content. Here we approximate that with
    ``skip_special_tokens=True``, which only matches exactly when the stop
    sequence itself is a special token -- a plain-string stop sequence will
    still show up in the decoded content.
    """
    if len(turn_record.segments) == 1 and turn_record.segments[0].raw_response is not None:
        raw = turn_record.segments[0].raw_response
        if turn_record.segments[0].replay_routed_experts is None:
            return raw
        choice = raw["choices"][0]
        meta_info = dict(choice["meta_info"])
        total_edges = len(turn_record.prompt_token_ids) + len(turn_record.output_token_ids) - 1
        _attach_replay(meta_info, turn_record, field="routed_experts", total_edges=total_edges)
        return {**raw, "choices": [{**choice, "meta_info": meta_info}, *raw["choices"][1:]]}

    token_ids = turn_record.output_token_ids
    logprobs = turn_record.output_logprobs
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    finish_reason = turn_record.finish_reason
    content = text
    tool_calls = None
    if tool_call_parser is not None and finish_reason == "stop":
        content, parsed_calls = tool_call_parser.parse_non_stream(text)
        if parsed_calls:
            finish_reason = "tool_calls"
            tool_calls = [
                {
                    "id": f"call{i:05d}",
                    "index": call.tool_index,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.parameters or "{}"},
                }
                for i, call in enumerate(parsed_calls)
            ]

    meta_info = {
        "output_token_logprobs": list(zip(logprobs, token_ids, strict=True)),
        "completion_tokens": len(token_ids),
    }
    _attach_engine_client_observability(meta_info, turn_record)
    _attach_spec_counters(meta_info, turn_record)

    total_edges = len(turn_record.prompt_token_ids) + len(token_ids) - 1
    _attach_replay(meta_info, turn_record, field="routed_experts", total_edges=total_edges)
    _attach_replay(meta_info, turn_record, field="indexer_topk", total_edges=total_edges)

    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
                "meta_info": meta_info,
            }
        ]
    }


def _attach_engine_client_observability(meta_info: dict, turn_record: TurnRecord) -> None:
    """Record the abort/resume history of *turn_record* onto *meta_info*.

    ``weight_version`` (singular, last segment only) is kept for backward
    compatibility with callers that only look at the final version. The full
    per-segment ``weight_versions`` list plus segment/resume counts are the
    only way downstream ``Sample``s (see ``Sample.update_from_meta_info`` /
    ``session.samples.merge._compute_sample_from_openai_record``) can tell
    that a turn was actually split and resumed across a weight update,
    rather than served by a single uninterrupted engine call.
    """
    weight_versions = turn_record.weight_versions
    if weight_versions:
        meta_info["weight_version"] = weight_versions[-1]
        meta_info["weight_versions"] = weight_versions
    meta_info["gateway_segments"] = len(turn_record.segments)
    meta_info["gateway_resumed"] = max(0, len(turn_record.segments) - 1)
    meta_info["gateway_processing_time"] = turn_record.processing_time


def _segment_raw_meta_info(segment: GenerationSegment) -> dict:
    raw = segment.raw_response
    if not raw:
        return {}
    if "meta_info" in raw:
        return raw["meta_info"] or {}
    choices = raw.get("choices") or []
    if choices:
        return choices[0].get("meta_info") or {}
    return {}


def _attach_spec_counters(meta_info: dict, turn_record: TurnRecord) -> None:
    """Re-attach per-request spec counters dropped by response reconstruction.

    Counters are summed across abort/resume segments. Rate/length averages
    are left off the reconstructed ``meta_info`` so callers recompute them
    from the summed integers.
    """
    totals: dict[str, int] = {}
    for segment in turn_record.segments:
        src = _segment_raw_meta_info(segment)
        for key in _SPEC_COUNTER_KEYS:
            if key in src and src[key] is not None:
                totals[key] = totals.get(key, 0) + src[key]
    meta_info.update(totals)


def _attach_replay(meta_info: dict, turn_record: TurnRecord, *, field: str, total_edges: int) -> None:
    ledger = ReplayLedger()
    for segment in turn_record.segments:
        slice_ = segment.replay_routed_experts if field == "routed_experts" else segment.replay_indexer_topk
        ledger.commit(slice_)
    if not ledger.is_active:
        return

    array = ledger.materialize(total_len=total_edges)
    # Must stay int32 on the wire: the unmodified downstream decoder
    # (generate_endpoint_utils._decode_topk_buffer) always reads int32.
    if field == "routed_experts" and array.dtype == np.uint8:
        missing = array == 255
        array = array.astype(np.int32)
        array[missing] = -1
    encoded, _dtype_name = encode_replay_buffer(array, dtype_mode="int32")
    meta_info[field] = encoded
    if field == "indexer_topk":
        meta_info["indexer_topk_num_layers"] = array.shape[1]


def validate_continuity(records: list) -> bool:
    """Whether a black-box session's ``SessionRecord`` list has no gaps in
    generation-turn coverage.

    ``SessionRecord`` has no explicit ``turn_index`` field yet, so this uses
    the record's position in the list as the turn-index proxy -- a
    single-point-of-change fallback that keeps working once an explicit
    field is added (only this function would need to switch its source).
    """
    for index, record in enumerate(records):
        meta_info = record.response.get("choices", [{}])[0].get("meta_info", {})
        if "weight_version" not in meta_info and "finish_reason" not in meta_info:
            continue
        declared_index = meta_info.get("turn_index")
        if declared_index is not None and declared_index != index:
            return False
    return True
