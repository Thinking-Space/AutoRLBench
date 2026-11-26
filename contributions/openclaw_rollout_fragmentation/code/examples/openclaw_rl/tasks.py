from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from slime.utils.types import Sample


WORKSPACE = PurePosixPath("/root/.openclaw/workspace")


@dataclass(frozen=True)
class ClawGymTask:
    task_dir: Path
    task_id: str
    user_query: str
    metadata: dict[str, Any]
    input_mount_dir: PurePosixPath

    @property
    def input_files_dir(self) -> Path:
        return self.task_dir / "input_files"

    @property
    def reward_dir(self) -> Path:
        return self.task_dir / "reward"


def load_task(sample: Sample) -> ClawGymTask:
    metadata = dict(sample.metadata or {})
    raw_task_dir = metadata.get("task_dir")
    if not raw_task_dir:
        raise ValueError("ClawGym sample metadata must include task_dir")

    task_dir = Path(raw_task_dir).expanduser().resolve()
    entry_path = task_dir / "data_entry.json"
    if not entry_path.is_file():
        raise FileNotFoundError(f"missing ClawGym data_entry.json: {entry_path}")
    with entry_path.open(encoding="utf-8") as fp:
        entry = json.load(fp)

    task_id = str(entry.get("task_id") or sample.label or task_dir.name)
    user_query = str(entry.get("user_query") or sample.prompt or "")
    if not user_query:
        raise ValueError(f"ClawGym task {task_id} has an empty user_query")

    input_mount = PurePosixPath(entry.get("input_mount_dir") or str(WORKSPACE))
    try:
        input_mount.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError(f"input_mount_dir must be under {WORKSPACE}, got {input_mount}") from exc

    input_files_dir = task_dir / "input_files"
    if "input_mount_dir" in entry and not input_files_dir.is_dir():
        raise FileNotFoundError(f"task declares input_mount_dir but has no input_files/: {task_dir}")
    if not (task_dir / "reward" / "reward.sh").is_file():
        raise FileNotFoundError(f"missing ClawGym reward/reward.sh: {task_dir}")

    entry_metadata = entry.get("metadata") or {}
    if not isinstance(entry_metadata, dict):
        raise TypeError(f"ClawGym task metadata must be an object: {entry_path}")
    return ClawGymTask(
        task_dir=task_dir,
        task_id=task_id,
        user_query=user_query,
        metadata=entry_metadata,
        input_mount_dir=input_mount,
    )


def discover_tasks(dataset_dir: Path) -> list[ClawGymTask]:
    tasks = []
    for entry_path in sorted(dataset_dir.glob("*/data_entry.json")):
        sample = Sample(metadata={"task_dir": str(entry_path.parent)})
        tasks.append(load_task(sample))
    if not tasks:
        raise ValueError(f"no ClawGym task directories found under {dataset_dir}")
    return tasks
