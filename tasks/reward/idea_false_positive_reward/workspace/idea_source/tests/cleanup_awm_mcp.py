#!/usr/bin/env python3
"""Kill leftover Agent World Model MCP processes.

Training and eval can leave uvicorn / ``server_runtime.py`` / ``awm.core.server``
and the SEED AWM bridge behind after an interrupt.  This script only matches
those leftovers for the current user; it does not touch Ray workers or the
trainer itself.

Examples:
    python3 tests/cleanup_awm_mcp.py
    python3 tests/cleanup_awm_mcp.py --kill
    python3 tests/cleanup_awm_mcp.py --kill --work-root outputs/awm_qwen35_rollouts_8_per_task
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


MATCH_NEEDLES = (
    "awm.core.server",
    "server_runtime.py",
    "agent_system/environments/env_package/awm/bridge.py",
)


@dataclass(frozen=True)
class Proc:
    pid: int
    etime: str
    cmd: str


def parse_ps(text: str) -> list[Proc]:
    procs: list[Proc] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_s, etime, cmd = parts
        if not pid_s.isdigit():
            continue
        procs.append(Proc(pid=int(pid_s), etime=etime, cmd=cmd))
    return procs


def list_user_procs() -> list[Proc]:
    out = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,etime=,cmd="],
        text=True,
    )
    return parse_ps(out)


def is_self(cmd: str) -> bool:
    return "cleanup_awm_mcp.py" in cmd


def is_mcp_leftover(cmd: str, work_root: str | None) -> bool:
    if is_self(cmd):
        return False
    if work_root and work_root not in cmd:
        return False
    if any(needle in cmd for needle in MATCH_NEEDLES):
        return True
    # Shell wrappers used by older AWM eval launches:
    #   sh -c '...server_runtime.py 2>&1 | tee .../server.log'
    return "tee " in cmd and "/server.log" in cmd and (
        "server_runtime.py" in cmd or "awm" in cmd.lower()
    )


def send_signal(pids: list[int], sig: signal.Signals) -> int:
    sent = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent += 1
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"skip pid {pid}: {exc}", file=sys.stderr)
    return sent


def still_alive(pids: list[int]) -> list[int]:
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            alive.append(pid)
    return alive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Send SIGTERM, then SIGKILL to matching processes. Default is dry-run.",
    )
    parser.add_argument(
        "--work-root",
        default=None,
        help="Only match processes whose command line contains this path.",
    )
    parser.add_argument(
        "--term-grace",
        type=float,
        default=2.0,
        help="Seconds to wait after SIGTERM before SIGKILL (default: 2).",
    )
    args = parser.parse_args()

    work_root = args.work_root
    if work_root:
        work_root = os.path.abspath(os.path.expanduser(work_root))

    matches = [proc for proc in list_user_procs() if is_mcp_leftover(proc.cmd, work_root)]
    if not matches:
        print("No leftover AWM MCP processes found.")
        return 0

    print(f"Matched {len(matches)} process(es):")
    for proc in matches:
        print(f"  pid={proc.pid:<8} etime={proc.etime:<12} {proc.cmd}")

    if not args.kill:
        print("Dry-run only. Re-run with --kill to terminate them.")
        return 0

    pids = [proc.pid for proc in matches]
    print(f"Sending SIGTERM to {len(pids)} process(es)...")
    send_signal(pids, signal.SIGTERM)
    time.sleep(max(0.0, args.term_grace))
    leftover = still_alive(pids)
    if leftover:
        print(f"Sending SIGKILL to {len(leftover)} remaining process(es)...")
        send_signal(leftover, signal.SIGKILL)
        time.sleep(0.5)

    remaining = [
        proc for proc in list_user_procs() if is_mcp_leftover(proc.cmd, work_root)
    ]
    if remaining:
        print(f"Still running after cleanup: {len(remaining)}", file=sys.stderr)
        for proc in remaining:
            print(f"  pid={proc.pid} {proc.cmd}", file=sys.stderr)
        return 1

    print("AWM MCP leftovers cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
