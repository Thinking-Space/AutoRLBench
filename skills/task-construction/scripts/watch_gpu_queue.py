#!/usr/bin/env python3
"""Wait without an LLM; submit a Codex continuation when the GPU queue is ready."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import gpu_queue as queue


def codex_command(value):
    if value:
        command = json.loads(value)
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("--codex-command-json must be a nonempty string array")
        return command
    executable = shutil.which("codex.exe") or shutil.which("codex")
    if not executable:
        raise FileNotFoundError("Codex CLI unavailable; supply --codex-command-json")
    path = Path(executable)
    if path.suffix.lower() in (".ps1", ".cmd", ".bat"):
        node = shutil.which("node")
        entry = path.parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        if not node or not entry.is_file():
            raise ValueError("Use a native Codex executable or a Node CLI command array")
        return [node, str(entry)]
    return [executable]


def poll(path):
    state, probed = queue.check_state(path)
    print(json.dumps({"probed": probed, **state}, ensure_ascii=False), flush=True)
    return state


def deliver(path, thread, command, cwd):
    state = queue.load(path)
    if state["status"] != "ready":
        raise ValueError("Queue is no longer ready")
    message = (
        f"已确认的实验算力队列现为 ready。排队状态文件：{path}。"
        "继续此 case 的实验准备：读取原方案和状态，"
        "执行 gpu_queue.py claim 重新确认，仅在本次新 claim 成功且团队资源分配允许时，"
        "按已有授权与预算启动阶段 4；已启动则核对真实作业，不重复启动。"
        "不要扩大执行范围或向远程 push。"
    )
    completed = subprocess.run(
        [*command, "queue", "--thread", thread, "--message", message],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Codex queue failed")


def watch(path, thread, command, cwd):
    state = queue.load(path)
    state["watcher"] = {"pid": os.getpid(), "thread": thread, "started_at": queue.iso(time.time())}
    queue.save(path, state)
    while True:
        state = queue.load(path)
        if state["status"] in ("claimed", "cancelled"):
            return {"status": state["status"], "woken": False}
        now = time.time()
        if state["status"] == "ready":
            deliver(path, thread, command, cwd)
            return {"status": "wake_delivered", "woken": True}
        if now >= state["next_check_epoch"]:
            poll(path)
            continue
        due = state["next_check_epoch"]
        # Local cancellation checks do not contact the GPU host or an LLM.
        time.sleep(max(0.1, min(60, due - now)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--thread", required=True, help="Explicit, reachable Codex session UUID; never --last")
    parser.add_argument("--codex-command-json", help="CLI argv prefix, e.g. [\"codex\"] or [\"node\",\"/path/codex.js\"]")
    parser.add_argument("--cwd", type=Path, help="Local workspace for the Codex CLI")
    args = parser.parse_args(argv)
    try:
        uuid.UUID(args.thread)
        path = args.state.expanduser().resolve(strict=True)
        cwd = args.cwd.expanduser().resolve(strict=True) if args.cwd else path.parent
        command = codex_command(args.codex_command_json)
        result = watch(path, args.thread, command, str(cwd))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
