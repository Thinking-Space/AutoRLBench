#!/usr/bin/env python3
"""Create one Tencent AGS sandbox and verify its task image starts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import tomllib
from pathlib import Path

from tencent_ags_e2b import build_create_kwargs, build_custom_config, resolve_task_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(__file__).with_name("harbor_tasks"),
    )
    parser.add_argument("--instance-id", default="")
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TENCENT_AGS_SMOKE_TIMEOUT", "300")),
    )
    return parser.parse_args()


def select_task_dir(tasks_dir: Path, instance_id: str) -> Path:
    if instance_id:
        task_dir = tasks_dir / instance_id
        if not task_dir.is_dir():
            raise FileNotFoundError(task_dir)
        return task_dir
    selected_ids = Path(__file__).with_name("data") / "selected_instance_ids.txt"
    first_id = selected_ids.read_text(encoding="utf-8").splitlines()[0].strip()
    task_dir = tasks_dir / first_id
    if not task_dir.is_dir():
        raise FileNotFoundError(task_dir)
    return task_dir


async def main() -> int:
    args = parse_args()
    if not os.environ.get("E2B_DOMAIN", "").strip() and os.environ.get("TENCENT_AGS_DOMAIN", "").strip():
        os.environ["E2B_DOMAIN"] = os.environ["TENCENT_AGS_DOMAIN"].strip()
    if not os.environ.get("E2B_API_URL", "").strip() and os.environ.get("TENCENT_AGS_URL", "").strip():
        os.environ["E2B_API_URL"] = os.environ["TENCENT_AGS_URL"].strip()
    if not os.environ.get("E2B_API_KEY", "").strip() and os.environ.get("E2B_API_KEY_FILE", "").strip():
        key_file = Path(os.environ["E2B_API_KEY_FILE"]).expanduser()
        os.environ["E2B_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
    if not os.environ.get("E2B_API_KEY", "").strip():
        raise RuntimeError("set E2B_API_KEY or let the launcher resolve E2B_API_KEY_FILE")
    if not os.environ.get("TENCENT_AGS_TEMPLATE", "").strip():
        raise RuntimeError("set TENCENT_AGS_TEMPLATE to the fixed AGS template")

    task_dir = select_task_dir(args.tasks_dir, args.instance_id)
    with (task_dir / "task.toml").open("rb") as task_file:
        task_config = tomllib.load(task_file)
    environment = task_config["environment"]
    image = resolve_task_image(environment.get("docker_image"), task_dir / "environment")
    metadata = {
        "environment_name": task_dir.name,
        "session_id": "tencent-ags-smoke",
        "purpose": "miles-harbor-mini-swe-smoke",
        "x-custom-config": json.dumps(
            build_custom_config(
                image,
                int(environment.get("cpus", 1)),
                int(environment.get("memory_mb", 4096)),
            ),
            separators=(",", ":"),
        ),
    }
    create_kwargs = build_create_kwargs(
        template=os.environ["TENCENT_AGS_TEMPLATE"],
        metadata=metadata,
        timeout=args.timeout,
    )
    print(f"[smoke] task={task_dir.name}")
    print(f"[smoke] image={image}")
    print(f"[smoke] template={os.environ['TENCENT_AGS_TEMPLATE']}")

    from e2b import AsyncSandbox

    sandbox = await AsyncSandbox.create(**create_kwargs)
    try:
        command = (
            "set -eu\n"
            "echo AGS_SMOKE_OK\n"
            "uname -a\n"
            "test -d /testbed\n"
            "python3 --version\n"
        )
        result = await sandbox.commands.run(command, timeout=60, user="root")
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.exit_code != 0:
            raise RuntimeError(f"AGS smoke command exited with {result.exit_code}")
        print(f"[smoke] sandbox_id={sandbox.sandbox_id}")
        return 0
    finally:
        try:
            await asyncio.wait_for(asyncio.shield(sandbox.kill()), timeout=20)
        except BaseException:
            print(
                f"[smoke] warning: could not kill sandbox {sandbox.sandbox_id}",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
