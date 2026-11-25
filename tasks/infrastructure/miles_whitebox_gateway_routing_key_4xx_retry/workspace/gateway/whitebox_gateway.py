"""White-box rollout gateway.

The rollout-facing side of a gateway boundary. Unlike a catch-all proxy, this
route answers through the shared HTTP retry client: it reads the caller's
request, drives the routing proxy itself, and renders the reply back.
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, Request

from miles.utils.http_utils import post

ROUTING_PROXY_URL = os.environ.get("GATEWAY_ROUTING_PROXY_URL", "http://127.0.0.1:30110")
LISTEN_PORT = int(os.environ.get("GATEWAY_PORT", "30100"))

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions/{session_id}/generate")
async def generate(session_id: str, request: Request):
    body = await request.json()
    output = await post(f"{ROUTING_PROXY_URL}/generate", body, headers=None)
    return output


if __name__ == "__main__":
    # Surface the retry client's per-attempt INFO lines (the response body of
    # every rejected request is the primary diagnostic this boundary emits).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
