"""Pluggable HTTP transport for engine calls -- the single toggle point for
the token-level partial-rollout gateway feature.

Default (``--rollout-gateway-transport local``, i.e. unset): returns the
exact same objects used before this feature existed
(``miles.utils.http_utils.post`` / a plain ``httpx.AsyncClient`` with
identical construction args), so a request never touches any new code and
stays byte-for-byte identical to today's behavior.
"""

import uuid

import httpx

from miles.utils.http_utils import post


def load_engine_transport(args):
    """Transport used by ``generate_hub/{single_turn,multi_turn}.py`` in
    place of calling ``miles.utils.http_utils.post`` directly."""
    if getattr(args, "rollout_gateway_transport", "local") != "remote":
        return post

    assert args.rollout_gateway_url, "--rollout-gateway-url is required when --rollout-gateway-transport=remote"
    gateway_url = args.rollout_gateway_url.rstrip("/")

    async def _remote_post(url, payload, max_retries=60, action="post", headers=None):
        del url  # the gateway always targets its own configured backend, not the caller's raw sglang URL
        headers = dict(headers or {})
        session_id = headers.get("X-SMG-Routing-Key") or str(uuid.uuid4())
        return await post(
            f"{gateway_url}/sessions/{session_id}/generate",
            payload,
            max_retries=max_retries,
            action=action,
            headers=headers,
        )

    return _remote_post


def create_engine_transport(args, backend_url: str, *, timeout: float):
    """Transport used by ``SessionServer`` (the black-box TITO proxy path).

    Remote mode is currently still a transparent httpx forward -- wiring the
    black-box path through ``EngineClient`` depends on the TITO/branch-manager
    owner's seam contract (design doc seam (2)), which isn't settled yet. This
    factory exists so that integration only needs to change this function,
    not any call site in ``session/server.py``.
    """
    if getattr(args, "rollout_gateway_transport", "local") != "remote":
        return httpx.AsyncClient(limits=httpx.Limits(max_connections=1024), timeout=httpx.Timeout(timeout))

    return httpx.AsyncClient(limits=httpx.Limits(max_connections=1024), timeout=httpx.Timeout(timeout))
