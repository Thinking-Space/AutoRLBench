"""Parses raw sglang ``/generate`` responses into EngineClient-internal segments.

The single place that knows the sglang response shape (``meta_info``
fields), so ``EngineClient`` and tests both interpret it the same way.
"""

from miles.rollout.engine_client.replay import decode_replay_buffer
from miles.rollout.engine_client.types import GenerationSegment, ReplaySlice
from miles.utils.replay_base import compact_routed_experts


def parse_meta_info(
    output: dict, *, args, path: str, prefix_len: int, elapsed_seconds: float = 0.0
) -> GenerationSegment:
    """Parse one raw engine response into a ``GenerationSegment``.

    ``path`` selects the response shape to expect: ``"generate"`` is the
    native sglang ``/generate`` shape (``output["meta_info"]``,
    ``finish_reason`` as ``{"type": ...}``); ``"chat_completions"`` is the
    OpenAI-compatible ``/v1/chat/completions`` shape
    (``output["choices"][0]["meta_info"]``, ``finish_reason`` as a bare
    string). Both shapes normalize to the same ``meta_info`` dict view
    before the rest of the parsing (including replay-slice decoding) runs
    identically.

    ``prefix_len`` is ``len(input_ids)`` as sent in *this* request (prompt
    plus whatever was already generated before this segment). The engine's
    routed-experts/indexer-topk buffers cover the full prefix generated so
    far, so the resulting ``ReplaySlice`` always starts at edge index 0 and
    ends at ``prefix_len + len(new_tokens) - 1`` -- the same
    ``len(tokens) - 1`` convention as ``Sample.rollout_routed_experts``.
    """
    assert path in ("generate", "chat_completions"), f"unknown path: {path}"
    if path == "generate":
        meta_info = output["meta_info"]
        finish_reason = meta_info["finish_reason"]["type"]
    else:
        choice = output["choices"][0]
        meta_info = choice["meta_info"]
        finish_reason = choice["finish_reason"]

    # sglang's native /generate path emits (logprob, token_id) pairs, while the
    # /v1/chat/completions path (return_text_in_logprobs=True) emits
    # (logprob, token_id, token_text) triples; only the first two fields matter here.
    logprob_pairs = meta_info.get("output_token_logprobs") or []
    token_ids = tuple(entry[1] for entry in logprob_pairs)
    logprobs = tuple(entry[0] for entry in logprob_pairs)
    num_edges = prefix_len + len(token_ids) - 1

    return GenerationSegment(
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason=finish_reason,
        weight_version=meta_info.get("weight_version"),
        replay_routed_experts=_parse_replay_slice(meta_info, field="routed_experts", num_edges=num_edges, args=args),
        replay_indexer_topk=_parse_replay_slice(meta_info, field="indexer_topk", num_edges=num_edges, args=args),
        raw_response=_without_routed_experts(output, path=path),
        elapsed_seconds=elapsed_seconds,
    )


def _parse_replay_slice(meta_info: dict, *, field: str, num_edges: int, args) -> ReplaySlice | None:
    info = meta_info.get(field)
    if info is None:
        return None

    if field == "routed_experts":
        array = decode_replay_buffer(info, num_tokens=num_edges, num_layers=args.num_layers, topk=args.moe_router_topk)
        array = compact_routed_experts(array, getattr(args, "num_experts", None))
    else:
        num_layers = meta_info.get("indexer_topk_num_layers")
        assert num_layers is not None, (
            "Server returned indexer_topk without indexer_topk_num_layers; "
            "sglang-miles must include the layer count in meta_info."
        )
        array = decode_replay_buffer(info, num_tokens=num_edges, num_layers=num_layers, topk=-1)

    return ReplaySlice(start=0, end=num_edges, array=array)


def _without_routed_experts(output: dict, *, path: str) -> dict:
    """Copy only the response containers owning the redundant wire payload."""
    meta_info = output["meta_info"] if path == "generate" else output["choices"][0]["meta_info"]
    if meta_info.get("routed_experts") is None:
        return output
    if path == "generate":
        meta = {key: value for key, value in output["meta_info"].items() if key != "routed_experts"}
        return {**output, "meta_info": meta}
    choice = output["choices"][0]
    meta = {key: value for key, value in choice["meta_info"].items() if key != "routed_experts"}
    return {**output, "choices": [{**choice, "meta_info": meta}, *output["choices"][1:]]}
