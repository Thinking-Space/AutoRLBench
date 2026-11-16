"""Sandbox lifecycle for agentic rollouts: admission, setup, tool calls, close."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sandbox_client import Sandbox

logger = logging.getLogger(__name__)

SANDBOX_OPERATION_CONCURRENCY = int(os.getenv("SANDBOX_OPERATION_CONCURRENCY", "32"))
MAX_CONCURRENT_SANDBOX_SETUP = int(os.getenv("MAX_CONCURRENT_SANDBOX_SETUP", str(SANDBOX_OPERATION_CONCURRENCY)))
SLOW_SEMAPHORE_WAIT_SECONDS = float(os.getenv("SLOW_SEMAPHORE_WAIT_SECONDS", "5"))
SANDBOX_TEMPLATE = os.getenv("SANDBOX_TEMPLATE", "python-repo")
SETUP_MAX_RETRIES = int(os.getenv("SANDBOX_SETUP_MAX_RETRIES", "3"))

_setup_semaphore: asyncio.Semaphore | None = None


def get_sandbox_setup_semaphore() -> asyncio.Semaphore:
    global _setup_semaphore
    if _setup_semaphore is None:
        _setup_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SANDBOX_SETUP)
    return _setup_semaphore


@dataclass
class SetupRecord:
    session: str
    phase: str = "waiting_for_semaphore"
    semaphore_wait_s: float = 0.0
    create_s: float = 0.0
    tool_setup_s: float = 0.0
    attempts: list = field(default_factory=list)
    sandbox_id: str | None = None
    status: str = "waiting"


def _emit(record: SetupRecord) -> None:
    root = os.environ.get("RUN_EVIDENCE_DIR")
    if not root:
        return
    path = Path(root) / "env_setup" / f"env_setup.{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": time.time(), **asdict(record)}, sort_keys=True) + "\n")


class SandboxEnv:
    def __init__(self, sandbox: Sandbox, record: SetupRecord):
        self.sandbox = sandbox
        self.record = record

    async def call(self, name: str, params: dict) -> str:
        return await self.sandbox.run_tool(name, params)

    async def close(self) -> None:
        try:
            await self.sandbox.kill()
        except Exception as exc:
            logger.error("failed to close sandbox %s: %r", self.sandbox.sandbox_id, exc)


async def create_and_setup_env(session: str, files: dict[str, str], test_failure: str) -> SandboxEnv:
    record = SetupRecord(session=session)
    semaphore = get_sandbox_setup_semaphore()
    wait_start = time.monotonic()
    async with semaphore:
        record.semaphore_wait_s = round(time.monotonic() - wait_start, 4)
        record.phase, record.status = "semaphore_acquired", "running"
        if record.semaphore_wait_s >= SLOW_SEMAPHORE_WAIT_SECONDS:
            logger.warning("sandbox setup waited %.1fs for semaphore", record.semaphore_wait_s)
        last_exception = None
        for attempt in range(SETUP_MAX_RETRIES):
            sandbox = None
            attempt_info = {"attempt": attempt + 1, "phase": "create_env"}
            record.attempts.append(attempt_info)
            try:
                started = time.monotonic()
                sandbox = await Sandbox.create(SANDBOX_TEMPLATE, metadata={"session": session})
                record.create_s = round(time.monotonic() - started, 4)
                record.sandbox_id = sandbox.sandbox_id
                attempt_info["phase"] = "tool_setup"
                started = time.monotonic()
                await sandbox.upload_files(files, test_failure)
                await sandbox.install_commands()
                record.tool_setup_s = round(time.monotonic() - started, 4)
                attempt_info["status"] = "completed"
                record.phase, record.status = "done", "completed"
                _emit(record)
                return SandboxEnv(sandbox, record)
            except Exception as exc:
                last_exception = exc
                attempt_info.update(status="failed", error=type(exc).__name__)
                logger.warning("create_and_setup_env failed (attempt %d/%d): %r", attempt + 1, SETUP_MAX_RETRIES, exc)
                if sandbox is not None:
                    try:
                        await sandbox.kill()
                    except Exception:
                        logger.warning("failed to close sandbox after setup failure")
                if attempt < SETUP_MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
        record.phase, record.status = record.attempts[-1].get("phase", "unknown"), "failed"
        _emit(record)
        raise last_exception
