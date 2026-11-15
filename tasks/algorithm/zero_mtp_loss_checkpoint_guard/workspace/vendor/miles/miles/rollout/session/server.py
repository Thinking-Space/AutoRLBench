"""Standalone session-server process: HTTP chassis + upstream proxy transport.

- ``SessionServer`` is a FastAPI app plus one shared httpx client; ``do_proxy`` forwards a request to the inference router (sglang or miles) — which does the load balancing to worker engines — and returns the raw result, or a 502 JSON error on transport failure.
- Session/TITO logic lives in ``core.SessionCore``; ``setup_session_routes`` (``sessions.py``) wires the HTTP routes to it.
- Standalone (own process, own event loop) so sessions also work with the SGLang Rust Router or any other backend, decoupled from the Miles Router.
- ``run_session_server`` is the subprocess entry point: fresh interpreter, so it configures logging and the process title itself, then serves uvicorn.
"""

import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI

from miles.rollout.engine_client.client import EngineClient
from miles.rollout.engine_client.transport import create_engine_transport
from miles.rollout.session.core import ProxyRequest
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.http_utils import filter_forwarded_headers, init_http_client, is_http_client_initialized
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        # This client owns only Session Server -> SGLang Router transport.
        # create_engine_transport is a seam for the partial-rollout gateway; in
        # both of its branches it currently returns the same httpx.AsyncClient
        # this used to construct inline, so routing through it changes nothing
        # today and keeps the integration point in one place.
        self.client = create_engine_transport(args, backend_url, timeout=timeout)

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        # run_session_server is launched via multiprocessing spawn (fresh
        # interpreter), so the parent process's init_http_client() call never
        # reaches this process's module-global _http_client -- do it here,
        # idempotently, before EngineClient's default post_fn needs it.
        if not is_http_client_initialized():
            init_http_client(args)
        engine_client_timeout = getattr(args, "engine_client_timeout", 600.0)
        self.engine_client = EngineClient(backend_url, args=args, timeout=engine_client_timeout)

        setup_session_routes(self.app, self, args)

    async def do_proxy(self, request: ProxyRequest, path: str, *, body: bytes, headers: dict) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"

        headers = filter_forwarded_headers(headers)

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("session_server")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
