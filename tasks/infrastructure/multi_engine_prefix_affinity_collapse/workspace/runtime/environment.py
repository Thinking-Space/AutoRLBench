"""Per-trajectory tool environment for the repository workload.

Tool results are computed from the repository carried in the sample metadata,
so they are deterministic and need no network. Each call still takes the time
a remote sandbox would need to run it (seeded, log-normal around a per-tool
median), so a trajectory spends wall time between engine calls the way an
agent with a real sandbox does.
"""
from __future__ import annotations

import asyncio
import os
import random

from . import repo_tools

# median seconds per tool call in a remote sandbox
TOOL_LATENCY_S = {"list_files": 0.4, "read_file": 0.6, "search": 1.0, "run_tests": 3.0, "submit_fix": 0.2}
LATENCY_SCALE = float(os.environ.get("TOOL_LATENCY_SCALE", "1.0"))


class EnvironmentError_(RuntimeError):
    """Raised when the environment cannot serve a tool call."""


class RepoEnvironment:
    def __init__(self, metadata: dict, session: str):
        self.files = metadata["files"]
        self.test_failure = metadata["test_failure"]
        self.rng = random.Random(f"{metadata.get('task_id')}:{session}")

    async def call(self, name: str, params: dict) -> str:
        median = TOOL_LATENCY_S.get(name, 0.5) * LATENCY_SCALE
        await asyncio.sleep(median * self.rng.lognormvariate(0.0, 0.4))
        return repo_tools.execute(self.files, self.test_failure, name, params)

    async def close(self) -> None:
        return None


async def open_environment(sample) -> RepoEnvironment:
    metadata = sample.metadata or {}
    session = sample.routing_key or str(sample.index)
    return RepoEnvironment(metadata, session)
