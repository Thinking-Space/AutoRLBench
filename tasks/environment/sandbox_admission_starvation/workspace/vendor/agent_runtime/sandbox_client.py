"""Async HTTP client for the sandbox control plane."""
from __future__ import annotations

import os

import httpx

SANDBOX_API_URL = os.getenv("SANDBOX_API_URL", "http://127.0.0.1:18080")
SANDBOX_MAX_CONNECTIONS = int(os.getenv("SANDBOX_MAX_CONNECTIONS", "20"))
SANDBOX_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("SANDBOX_MAX_KEEPALIVE_CONNECTIONS", "20"))
SANDBOX_KEEPALIVE_EXPIRY = float(os.getenv("SANDBOX_KEEPALIVE_EXPIRY", "30"))
SANDBOX_REQUEST_TIMEOUT = float(os.getenv("SANDBOX_REQUEST_TIMEOUT", "60"))
SANDBOX_POOL_TIMEOUT = float(os.getenv("SANDBOX_POOL_TIMEOUT", "30"))

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Process-wide client; every sandbox operation shares this connection pool."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=SANDBOX_API_URL,
            limits=httpx.Limits(
                max_connections=SANDBOX_MAX_CONNECTIONS,
                max_keepalive_connections=SANDBOX_MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=SANDBOX_KEEPALIVE_EXPIRY,
            ),
            timeout=httpx.Timeout(SANDBOX_REQUEST_TIMEOUT, pool=SANDBOX_POOL_TIMEOUT),
            trust_env=False,
        )
    return _client


class Sandbox:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id

    @classmethod
    async def create(cls, template: str, metadata: dict | None = None) -> "Sandbox":
        response = await get_client().post("/sandboxes", json={"template": template, "metadata": metadata or {}})
        response.raise_for_status()
        return cls(response.json()["sandbox_id"])

    async def upload_files(self, files: dict[str, str], test_failure: str) -> None:
        response = await get_client().post(f"/sandboxes/{self.sandbox_id}/files",
                                           json={"files": files, "test_failure": test_failure})
        response.raise_for_status()

    async def install_commands(self) -> None:
        response = await get_client().post(f"/sandboxes/{self.sandbox_id}/commands", json={})
        response.raise_for_status()

    async def run_tool(self, tool: str, params: dict) -> str:
        response = await get_client().post(f"/sandboxes/{self.sandbox_id}/exec", json={"tool": tool, "params": params})
        response.raise_for_status()
        return response.json()["output"]

    async def kill(self) -> None:
        response = await get_client().delete(f"/sandboxes/{self.sandbox_id}")
        response.raise_for_status()
