"""In-process fake sglang ``/generate`` endpoint for EngineClient unit tests.

Lets tests script a sequence of responses (e.g. abort after N tokens, then
stop with the remainder) without a real inference engine or network. Matches
``miles.utils.http_utils.post``'s signature so it can be injected as
``EngineClient(..., post_fn=fake_engine.post)``.
"""

from dataclasses import dataclass


@dataclass
class ScriptedResponse:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str  # "abort" | "stop" | "length"
    weight_version: str | None = None
    routed_experts_b64: str | None = None
    indexer_topk_b64: str | None = None
    indexer_topk_num_layers: int | None = None
    response_shape: str = "generate"  # "generate" | "chat_completions"


class FakeEngine:
    """Replays a fixed script of responses, one per call, in order."""

    def __init__(self, script: list[ScriptedResponse]):
        self._script = list(script)
        self.calls: list[dict] = []

    async def post(self, url, payload, max_retries=60, action="post", headers=None):
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        assert self._script, "FakeEngine script exhausted"
        response = self._script.pop(0)

        meta_info = {
            "output_token_logprobs": list(zip(response.logprobs, response.token_ids, strict=True)),
            "completion_tokens": len(response.token_ids),
        }
        if response.response_shape == "generate":
            meta_info["finish_reason"] = {"type": response.finish_reason}
        if response.weight_version is not None:
            meta_info["weight_version"] = response.weight_version
        if response.routed_experts_b64 is not None:
            meta_info["routed_experts"] = response.routed_experts_b64
        if response.indexer_topk_b64 is not None:
            meta_info["indexer_topk"] = response.indexer_topk_b64
            meta_info["indexer_topk_num_layers"] = response.indexer_topk_num_layers

        if response.response_shape == "generate":
            return {"text": "", "meta_info": meta_info}
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": response.finish_reason,
                    "meta_info": meta_info,
                }
            ]
        }
