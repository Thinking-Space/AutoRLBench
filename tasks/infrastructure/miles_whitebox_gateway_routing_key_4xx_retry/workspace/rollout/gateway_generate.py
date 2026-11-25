"""Single-turn generation driven through the white-box gateway.

Same shape as the stock single-turn generate, but each sample's request goes
to the gateway's per-session generate route instead of the routing proxy
directly, carrying the sample's routing headers.
"""

import os

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:30100")


async def generate(input: GenerateFnInput, url: str | None = None) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"
    url = url or GATEWAY_URL

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)

    # Handle Partial Rollout resuming
    if len(sample.response) > 0:
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
        evaluation=input.evaluation,
        kv_cache_namespace=sample.kv_cache_namespace,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    session_id = sample.metadata.setdefault("gateway_session_id", f"gw-{sample.index:06d}")
    output = await post(
        f"{url}/sessions/{session_id}/generate",
        payload,
        headers=compute_routing_headers(args, sample),
    )
    await update_sample_from_response(args, sample, payload=payload, output=output)

    return GenerateFnOutput(samples=sample)
