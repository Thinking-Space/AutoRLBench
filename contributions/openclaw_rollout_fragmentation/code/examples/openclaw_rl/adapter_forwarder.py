"""Transparent worker-side HTTP forwarder for the stateful OpenClaw adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import time
import uuid
from collections.abc import Mapping
from typing import Any

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

logger = logging.getLogger("openclaw_adapter_forwarder")

FORWARDER_LIVENESS_PATH = "/__forwarder/live"
_CLIENT_KEY = web.AppKey("adapter_forwarder_client", aiohttp.ClientSession)
_REQUEST_ID_HEADER = "X-Request-ID"
_FORWARDER_NODE_HEADER = "X-Forwarder-Node"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _connection_header_tokens(headers: Mapping[str, str]) -> set[str]:
    value = headers.get("Connection", "")
    return {token.strip().lower() for token in value.split(",") if token.strip()}


def _forward_headers(headers: Mapping[str, str]) -> CIMultiDict[str]:
    blocked = _HOP_BY_HOP_HEADERS | _connection_header_tokens(headers) | {"host", "content-length"}
    return CIMultiDict((name, value) for name, value in headers.items() if name.lower() not in blocked)


def _response_headers(headers: Mapping[str, str]) -> CIMultiDict[str]:
    blocked = _HOP_BY_HOP_HEADERS | _connection_header_tokens(headers)
    return CIMultiDict((name, value) for name, value in headers.items() if name.lower() not in blocked)


def _session_id(body: bytes, headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return token
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    for key in ("session_id", "user"):
        if payload.get(key):
            return str(payload[key])
    return None


def _upstream_base_url(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{rendered_host}:{port}"


def _log_request(**fields: Any) -> None:
    logger.info("[adapter-forwarder] %s", json.dumps(fields, sort_keys=True, separators=(",", ":")))


def create_app(
    upstream_host: str,
    upstream_port: int,
    *,
    connect_timeout_sec: float = 5.0,
    forwarder_node: str | None = None,
) -> web.Application:
    upstream_base = _upstream_base_url(upstream_host, upstream_port)
    node = forwarder_node or socket.gethostname()
    app = web.Application(client_max_size=64 * 1024 * 1024)

    trace_config = aiohttp.TraceConfig()

    async def connection_start(_session, trace_ctx, _params) -> None:
        trace_ctx.trace_request_ctx["connect_started"] = time.monotonic()

    async def connection_end(_session, trace_ctx, _params) -> None:
        started = trace_ctx.trace_request_ctx.get("connect_started")
        if started is not None:
            trace_ctx.trace_request_ctx["connect_ms"] = round((time.monotonic() - started) * 1000)

    async def connection_reused(_session, trace_ctx, _params) -> None:
        trace_ctx.trace_request_ctx["connect_ms"] = 0

    trace_config.on_connection_create_start.append(connection_start)
    trace_config.on_connection_create_end.append(connection_end)
    trace_config.on_connection_reuseconn.append(connection_reused)

    async def client_context(application: web.Application):
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout_sec,
            sock_connect=connect_timeout_sec,
            sock_read=None,
        )
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        application[_CLIENT_KEY] = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            auto_decompress=False,
            trust_env=False,
            trace_configs=[trace_config],
        )
        try:
            yield
        finally:
            await application[_CLIENT_KEY].close()

    app.cleanup_ctx.append(client_context)

    async def liveness(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "forwarder_node": node,
                "upstream": upstream_base,
            }
        )

    async def proxy(request: web.Request) -> web.StreamResponse:
        started = time.monotonic()
        body = await request.read()
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        sid = _session_id(body, request.headers)
        target = f"{upstream_base}{request.raw_path}"
        headers = _forward_headers(request.headers)
        headers[_REQUEST_ID_HEADER] = request_id
        headers[_FORWARDER_NODE_HEADER] = node
        if request.remote:
            previous = request.headers.get("X-Forwarded-For")
            headers["X-Forwarded-For"] = f"{previous}, {request.remote}" if previous else request.remote
        headers["X-Forwarded-Proto"] = request.scheme

        timings: dict[str, Any] = {}
        downstream: web.StreamResponse | None = None
        status = 502
        response_bytes = 0
        error: str | None = None
        first_byte_ms: int | None = None
        try:
            client = request.app[_CLIENT_KEY]
            async with client.request(
                request.method,
                target,
                headers=headers,
                data=body,
                allow_redirects=False,
                trace_request_ctx=timings,
            ) as upstream:
                status = upstream.status
                downstream = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=_response_headers(upstream.headers),
                )
                await downstream.prepare(request)
                async for chunk in upstream.content.iter_chunked(64 * 1024):
                    if first_byte_ms is None:
                        first_byte_ms = round((time.monotonic() - started) * 1000)
                    response_bytes += len(chunk)
                    await downstream.write(chunk)
                await downstream.write_eof()
                return downstream
        except asyncio.CancelledError:
            error = "client_cancelled"
            raise
        except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError) as exc:
            status = 503
            error = f"{type(exc).__name__}:{exc}"
            return web.json_response(
                {"error": "adapter upstream unavailable", "request_id": request_id},
                status=status,
                headers={_REQUEST_ID_HEADER: request_id},
            )
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError) as exc:
            status = 502
            error = f"{type(exc).__name__}:{exc}"
            if downstream is not None and downstream.prepared:
                raise
            return web.json_response(
                {"error": "adapter upstream connection failed", "request_id": request_id},
                status=status,
                headers={_REQUEST_ID_HEADER: request_id},
            )
        except Exception as exc:
            status = 502
            error = f"{type(exc).__name__}:{exc}"
            logger.exception("[adapter-forwarder] request failed request_id=%s", request_id)
            if downstream is not None and downstream.prepared:
                raise
            return web.json_response(
                {"error": "adapter forwarding failed", "request_id": request_id},
                status=status,
                headers={_REQUEST_ID_HEADER: request_id},
            )
        finally:
            _log_request(
                request_id=request_id,
                session_id=sid,
                forwarder_node=node,
                upstream=f"{upstream_host}:{upstream_port}",
                method=request.method,
                path=request.raw_path,
                status=status,
                connect_ms=timings.get("connect_ms"),
                first_byte_ms=first_byte_ms,
                total_ms=round((time.monotonic() - started) * 1000),
                response_bytes=response_bytes,
                error=error,
            )

    app.router.add_get(FORWARDER_LIVENESS_PATH, liveness)
    app.router.add_route("*", "/{path:.*}", proxy)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8666)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, default=8666)
    parser.add_argument("--connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--forwarder-node", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app(
        args.upstream_host,
        args.upstream_port,
        connect_timeout_sec=args.connect_timeout_sec,
        forwarder_node=args.forwarder_node,
    )
    logger.info(
        "[adapter-forwarder] listening=%s:%d upstream=%s:%d",
        args.listen_host,
        args.listen_port,
        args.upstream_host,
        args.upstream_port,
    )
    web.run_app(
        app,
        host=args.listen_host,
        port=args.listen_port,
        print=None,
        handler_cancellation=True,
        reuse_port=False,
    )


if __name__ == "__main__":
    main()
