"""Trainer-side client for the session servers' ``/admin`` endpoints.

A session server can stop accepting new generations and wait for the in-flight
ones to land (``POST /admin/quiesce``), then start accepting again
(``POST /admin/resume``). Without this, a weight update can land in the middle of
a turn: the agent's next request is served by the new weights while the tokens
already recorded for that turn came from the old ones, and the resulting sample
carries log-probabilities from two policies with nothing marking the seam.

Quiescing is per process -- each session server counts only its own in-flight
turns -- so the trainer calls every instance. The endpoint list comes from
:func:`miles.ray.rollout.router_manager.session_server_endpoints`; it is passed
in rather than imported here, so this module stays on the session side of the
dependency edge (``router_manager`` already imports the session server).

Off unless ``--session-quiesce-on-weight-update`` is set. When off, every
function here returns before touching the network, so a run that does not ask
for this behaves exactly as it did before.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from miles.rollout.session.auth import AUTH_MODE_ENFORCE
from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

# Enough to ride out a transient connection failure, few enough that a genuinely
# unreachable server is reported quickly. The training step waits on this.
_MAX_RETRIES = 3

SessionServerEndpoints = list[tuple[str, int, str | None]]


def quiesce_enabled(args) -> bool:
    return bool(getattr(args, "session_quiesce_on_weight_update", False))


def admin_headers(args) -> dict[str, str]:
    """The admin key, or a precise complaint about why there is not one.

    The ``/admin`` routes only exist under ``--session-auth-mode enforce``, and
    an omitted ``--session-admin-key`` makes the server mint a random key that
    only it knows. Both mistakes would otherwise surface as a 404 or a 401 in
    the middle of a training step, a long way from the actual cause.
    """
    if getattr(args, "session_auth_mode", "off") != AUTH_MODE_ENFORCE:
        raise ValueError(
            "--session-quiesce-on-weight-update requires --session-auth-mode enforce: "
            "the /admin endpoints are not registered under 'off'"
        )
    admin_key = getattr(args, "session_admin_key", None)
    if not admin_key:
        raise ValueError(
            "--session-quiesce-on-weight-update requires an explicit --session-admin-key: "
            "without one the server mints a random key that only the server knows"
        )
    return {"Authorization": f"Bearer {admin_key}"}


async def _call_each(args, endpoints: SessionServerEndpoints, path: str, payload: dict) -> list[dict]:
    if not endpoints:
        return []
    headers = admin_headers(args)
    results = await asyncio.gather(
        *(
            post(f"http://{ip}:{port}{path}", payload, max_retries=_MAX_RETRIES, headers=headers)
            for ip, port, _ in endpoints
        ),
        return_exceptions=True,
    )
    reports: list[dict] = []
    for (ip, port, instance_id), result in zip(endpoints, results, strict=True):
        if isinstance(result, BaseException):
            # Report and carry on. Refusing to update weights because one session
            # server is unreachable turns a degraded instance into a stalled run;
            # the log names the instance that was skipped.
            logger.warning("[session] %s failed for %s:%s (%s): %s", path, ip, port, instance_id, result)
            reports.append({"endpoint": f"{ip}:{port}", "instance_id": instance_id, "error": str(result)})
            continue
        reports.append({"endpoint": f"{ip}:{port}", "instance_id": instance_id, **result})
    return reports


async def quiesce_session_servers(args, endpoints: SessionServerEndpoints, timeout: float = 300.0) -> list[dict]:
    """Stop admitting turns and wait for the in-flight ones, on every instance.

    A server that runs out of time says so (``timed_out: true``) instead of
    hanging. That is logged and the caller continues: the weight update is what
    the trainer is actually blocked on, and a turn that outlives the timeout has
    already lost its guarantee.
    """
    if not quiesce_enabled(args):
        return []
    reports = await _call_each(args, endpoints, "/admin/quiesce", {"timeout": timeout})
    for report in reports:
        if report.get("timed_out"):
            logger.warning(
                "[session] %s still had %s in-flight generation(s) after %ss; updating weights anyway",
                report.get("endpoint"),
                report.get("inflight_generations"),
                timeout,
            )
    return reports


async def resume_session_servers(args, endpoints: SessionServerEndpoints) -> list[dict]:
    """Start admitting turns again. Safe to call after a failed quiesce."""
    if not quiesce_enabled(args):
        return []
    return await _call_each(args, endpoints, "/admin/resume", {})


@asynccontextmanager
async def session_servers_quiesced(args, endpoints: SessionServerEndpoints, timeout: float = 300.0):
    """Quiesce for the duration of the block, and resume whatever happens.

    ``resume`` runs in a ``finally`` on purpose: a weight update that raises
    would otherwise leave every session server refusing generations for the rest
    of the run, which looks like a hang rather than an error.
    """
    await quiesce_session_servers(args, endpoints, timeout)
    try:
        yield
    finally:
        await resume_session_servers(args, endpoints)
