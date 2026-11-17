"""Run one grading command and hand back its log.

Grading runs are untrusted: the command executes code the policy just wrote, in the policy's
own checkout. It gets a wall-clock budget and its output is captured whole; nothing here
inspects the log.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

from .types import RunOutput

DEFAULT_TIMEOUT_S = 900


def run_command(argv: Sequence[str], cwd: Path, timeout_s: int = DEFAULT_TIMEOUT_S) -> RunOutput:
    env = dict(os.environ)
    # Keep the policy's environment out of the grading run, and keep the run deterministic.
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    started = time.time()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return RunOutput(command=" ".join(argv), exit_code=-1, log="", seconds=time.time() - started, timed_out=True)
    log = completed.stdout.decode("utf-8", errors="replace")
    return RunOutput(
        command=" ".join(argv),
        exit_code=completed.returncode,
        log=log,
        seconds=time.time() - started,
    )
