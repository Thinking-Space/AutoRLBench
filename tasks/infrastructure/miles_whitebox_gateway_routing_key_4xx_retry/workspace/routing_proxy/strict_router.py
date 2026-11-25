"""Routing-policy enforcement proxy in front of the rollout engine.

Stands where a consistent-hash routing proxy stands: every request must carry
the routing key header, and traffic without it is rejected with 400 before it
reaches the backend. Requests that do carry the key are forwarded upstream
with it attached.
"""

import json
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

UPSTREAM = os.environ.get("ROUTER_SHIM_UPSTREAM", "http://127.0.0.1:30210")
LISTEN_PORT = int(os.environ.get("ROUTER_SHIM_PORT", "30110"))
ROUTING_KEY_HEADER = "x-smg-routing-key"

# Headers that describe the inbound hop, not the one being forwarded.
_HOP_BY_HOP = ("content-length", "transfer-encoding", "host", "connection")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    if request.headers.get(ROUTING_KEY_HEADER) is None:
        body = {
            "error": {
                "code": "missing_routing_key",
                "message": (
                    "this routing policy routes every request by its "
                    "X-SMG-Routing-Key header; send the header with each request"
                ),
            }
        }
        return Response(content=json.dumps(body), status_code=400, media_type="application/json")

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    upstream_response = await request.app.state.client.post(f"{UPSTREAM}/generate", content=await request.body(), headers=headers)
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
