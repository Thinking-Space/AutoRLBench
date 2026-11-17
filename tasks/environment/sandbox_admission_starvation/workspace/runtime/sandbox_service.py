"""Local sandbox control plane used by the public rollout.

The service models the remote sandbox provider the original run used: creating
a sandbox, uploading the repository, installing tool commands, executing tool
calls and deleting the sandbox are separate HTTP operations with their own
latency.  Provisioning slows down as more sandboxes are being created at once,
which is how the hosted provider behaved under burst load.  All latencies come
from a seeded generator and every request is appended to a JSONL access log.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import repo_tools  # noqa: E402


class SandboxService:
    def __init__(self, options):
        self.options = options
        self.rng = random.Random(options.seed)
        self.ids = itertools.count(1)
        self.sandboxes: dict[str, dict] = {}
        self.provisioning = 0
        self.counters = {"created": 0, "deleted": 0, "exec": 0, "uploads": 0, "installs": 0}
        self.log = open(options.access_log, "a", buffering=1, encoding="utf-8")

    def _record(self, op: str, started: float, **extra) -> None:
        self.log.write(json.dumps({"op": op, "t": started, "duration_s": round(time.time() - started, 4),
                                   "live": len(self.sandboxes), "provisioning": self.provisioning, **extra}) + "\n")

    def _latency(self, median: float) -> float:
        return median * self.rng.lognormvariate(0.0, self.options.sigma)

    async def create(self, request: web.Request) -> web.Response:
        started = time.time()
        body = await request.json()
        self.provisioning += 1
        try:
            contention = 1.0 + self.provisioning / self.options.provider_capacity
            await asyncio.sleep(self._latency(self.options.create_s) * contention)
        finally:
            self.provisioning -= 1
        sandbox_id = f"sbx-{next(self.ids):06d}"
        self.sandboxes[sandbox_id] = {"template": body.get("template"), "files": {}, "test_failure": "", "commands": False}
        self.counters["created"] += 1
        self._record("create", started, sandbox=sandbox_id)
        return web.json_response({"sandbox_id": sandbox_id, "status": "running"})

    def _get(self, request: web.Request) -> dict:
        sandbox = self.sandboxes.get(request.match_info["sid"])
        if sandbox is None:
            raise web.HTTPNotFound(text="unknown sandbox")
        return sandbox

    async def upload(self, request: web.Request) -> web.Response:
        started = time.time()
        sandbox = self._get(request)
        body = await request.json()
        size = sum(len(v) for v in body.get("files", {}).values())
        await asyncio.sleep(self._latency(self.options.upload_s) * (1 + size / 200_000))
        sandbox["files"] = body.get("files", {})
        sandbox["test_failure"] = body.get("test_failure", "")
        self.counters["uploads"] += 1
        self._record("upload", started, sandbox=request.match_info["sid"], bytes=size)
        return web.json_response({"ok": True})

    async def install(self, request: web.Request) -> web.Response:
        started = time.time()
        sandbox = self._get(request)
        await asyncio.sleep(self._latency(self.options.install_s))
        sandbox["commands"] = True
        self.counters["installs"] += 1
        self._record("install", started, sandbox=request.match_info["sid"])
        return web.json_response({"ok": True})

    async def execute(self, request: web.Request) -> web.Response:
        started = time.time()
        sandbox = self._get(request)
        if not sandbox["commands"]:
            return web.json_response({"error": "tool commands are not installed"}, status=409)
        body = await request.json()
        await asyncio.sleep(self._latency(self.options.exec_s))
        output = repo_tools.execute(sandbox["files"], sandbox["test_failure"], body["tool"], body.get("params") or {})
        self.counters["exec"] += 1
        self._record("exec", started, sandbox=request.match_info["sid"], tool=body["tool"])
        return web.json_response({"output": output})

    async def delete(self, request: web.Request) -> web.Response:
        started = time.time()
        self.sandboxes.pop(request.match_info["sid"], None)
        await asyncio.sleep(self._latency(self.options.delete_s))
        self.counters["deleted"] += 1
        self._record("delete", started, sandbox=request.match_info["sid"])
        return web.json_response({"ok": True})

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response({"live": len(self.sandboxes), "provisioning": self.provisioning, **self.counters})


def build_app(options) -> web.Application:
    service = SandboxService(options)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([
        web.post("/sandboxes", service.create),
        web.post("/sandboxes/{sid}/files", service.upload),
        web.post("/sandboxes/{sid}/commands", service.install),
        web.post("/sandboxes/{sid}/exec", service.execute),
        web.delete("/sandboxes/{sid}", service.delete),
        web.get("/stats", service.stats),
        web.get("/health", lambda request: web.json_response({"ok": True})),
    ])
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SANDBOX_SERVICE_PORT", "18080")))
    parser.add_argument("--access-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--create-s", type=float, default=6.0)
    parser.add_argument("--upload-s", type=float, default=1.5)
    parser.add_argument("--install-s", type=float, default=2.0)
    parser.add_argument("--exec-s", type=float, default=0.4)
    parser.add_argument("--delete-s", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.45)
    parser.add_argument("--provider-capacity", type=float, default=48.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    opts = parse_args()
    opts.access_log.parent.mkdir(parents=True, exist_ok=True)
    web.run_app(build_app(opts), host=opts.host, port=opts.port, access_log=None, print=None)
