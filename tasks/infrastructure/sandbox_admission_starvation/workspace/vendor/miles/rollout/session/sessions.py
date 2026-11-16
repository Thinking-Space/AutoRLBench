"""Single-process FastAPI adapter for the session server.

Thin layer: converts each HTTP request to primitive inputs, calls
``SessionCore``. All session/TITO logic lives in ``core``.
"""

import json
import logging
from collections.abc import Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

from miles.rollout.engine_client.client import create_engine_client_router
from miles.rollout.session.auth import AUTH_MODE_OFF, SessionAuth
from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import MessageValidationError, SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.provider import (
    ProviderConfigurationError,
    SessionProfile,
    SessionProviderRegistry,
)
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.misc import load_function
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def _register_extension_routes(app, *, core, auth, args) -> None:
    """Mount routes named by ``--session-route-paths``.

    The session server owns sessions, turns and tokenization. Serving a wire
    protocol that is not chat -- OpenAI Responses, Anthropic Messages -- means
    translating it into a chat turn and the reply back again, and that
    translation is downstream policy, not session state. This hook is how it gets
    mounted without miles carrying a codec for every protocol somebody's agent
    happens to speak.

    Same shape as ``--session-provider-paths``: a mapping of label to dotted
    path, nothing imported unless listed, and a bad entry fails startup rather
    than the first request that would have used it.

    ``core`` and ``auth`` are passed because an extension route has to do what
    the built-in ones do -- check the session credential, then run the turn
    through ``core`` so it is recorded. A route that reached SGLang by itself
    would produce a session whose trajectory is missing exactly the turns it
    served.
    """
    route_paths = getattr(args, "session_route_paths", None) or {}
    if not isinstance(route_paths, Mapping):
        raise ValueError("session_route_paths must be a mapping of labels to dotted paths")
    for label, dotted_path in route_paths.items():
        try:
            register = load_function(str(dotted_path))
        except Exception as exc:  # noqa: BLE001 - any import failure is the same failure
            raise ProviderConfigurationError(
                f"failed to load session route extension {label!r} from {dotted_path!r}: {exc}"
            ) from exc
        register(app, core=core, auth=auth, args=args)
        logger.info("[session] registered route extension %s from %s", label, dotted_path)


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )

    provider_registry = SessionProviderRegistry()
    provider_paths = getattr(args, "session_provider_paths", None) or {}
    if not isinstance(provider_paths, Mapping):
        raise ValueError("session_provider_paths must be a mapping of profile names to dotted paths")
    for profile_name, dotted_path in provider_paths.items():
        provider_registry.register_dotted_path(str(profile_name), str(dotted_path))

    registry = SessionRegistry(
        args,
        tokenizer,
        tito_tokenizer=tito_tokenizer,
        provider_registry=provider_registry,
    )
    # engine_client is what routes a turn through the partial-rollout gateway
    # when --rollout-partial-mode=token; SessionCore falls back to the plain
    # proxy whenever it is None or the mode is off, so passing it costs nothing
    # for the providers that never opt in.
    core = SessionCore(backend, registry, args, session_server_instance_id, engine_client=backend.engine_client)
    auth = SessionAuth(
        mode=getattr(args, "session_auth_mode", AUTH_MODE_OFF) or AUTH_MODE_OFF,
        admin_key=getattr(args, "session_admin_key", None),
    )

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.get("/health")
    async def health():
        return await core.health()

    @app.get("/capabilities")
    async def capabilities(profile: str | None = None):
        return await core.capabilities(profile)

    @app.post("/sessions")
    async def create_session(request: Request):
        auth.check_admin(request)
        raw_body = await request.body()
        if not raw_body:
            data = {}
        else:
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise MessageValidationError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise MessageValidationError("session create body must be a JSON object")
        profile_explicit = "profile" in data or "metadata" in data
        profile_name = data.get("profile", "linear")
        metadata = data.get("metadata", {})
        if not isinstance(profile_name, str) or not isinstance(metadata, dict):
            raise MessageValidationError("profile must be a string and metadata an object")
        profile = SessionProfile(name=profile_name, metadata=metadata) if profile_explicit else None
        return await core.create_session(profile)

    @app.get("/sessions/{session_id}")
    async def get_session(request: Request, session_id: str, view: str | None = None):
        auth.check_admin(request)
        return await core.get_session(session_id, view=view)

    @app.delete("/sessions/{session_id}")
    async def delete_session(request: Request, session_id: str):
        auth.check_admin(request)
        return await core.delete_session(session_id)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        auth.check_session(request, core.session_token(session_id))
        body = await request.body()
        return await core.chat_completions(
            session_id,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )

    # ---------------------------------------------------------------------
    # Everything from here to the catch-all MUST stay above it. Starlette's
    # `{path:path}` converter swallows slashes, and FastAPI matches routes in
    # registration order, so a route registered after the catch-all is never
    # reached — and nothing reports that. That applies to the extension hook
    # below as well, which is where the non-chat protocols now live. The
    # ordering test in tests/fast/rollout/session/test_route_order.py guards it.
    # ---------------------------------------------------------------------

    @app.get("/sessions/{session_id}/v1/models")
    async def models(request: Request, session_id: str):
        auth.check_session(request, core.session_token(session_id))
        return await core.models(session_id)

    @app.post("/sessions/{session_id}/export-ack")
    async def export_ack(request: Request, session_id: str):
        auth.check_admin(request)
        raw_body = await request.body()
        try:
            data = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError as exc:
            raise MessageValidationError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise MessageValidationError("export-ack body must be a JSON object")
        return await core.acknowledge_export(session_id, data)

    # The partial-rollout gateway's own routes. Same ordering constraint as
    # everything else in this block -- registered before the catch-all, or
    # `{path:path}` swallows them.
    app.include_router(create_engine_client_router(backend, args, tokenizer=tokenizer))

    # Extension routes, registered here for one reason: this is the last point
    # still above the catch-all. See the ordering note above -- a route mounted
    # after `{path:path}` is silently unreachable, and an extension has no way to
    # know it landed in the dead zone.
    _register_extension_routes(app, core=core, auth=auth, args=args)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        auth.check_session(request, core.session_token(session_id))
        body = await request.body()
        return await core.proxy(
            session_id,
            path,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )

    # /admin/* is outside the /sessions/ prefix, so the catch-all above cannot
    # shadow it and registration order does not matter here. These are only
    # registered under `enforce`: an unauthenticated endpoint that stops the
    # server from accepting work is not something to leave lying around.
    if auth.enabled:

        @app.post("/admin/quiesce")
        async def quiesce(request: Request):
            auth.check_admin(request)
            raw_body = await request.body()
            try:
                data = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError as exc:
                raise MessageValidationError(f"invalid JSON body: {exc}") from exc
            return await core.quiesce(float(data.get("timeout", 300)))

        @app.post("/admin/resume")
        async def resume(request: Request):
            auth.check_admin(request)
            return await core.resume()

        @app.get("/admin/drain")
        async def drain(request: Request):
            auth.check_admin(request)
            return core.drain_status()
    else:
        logger.info("[session] Admin endpoints not registered (--session-auth-mode is 'off').")
