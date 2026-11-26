"""Prepare the local ClawGym RM tasks for the OpenClaw E2B evaluator.

The evaluator normally embeds every input in JSON.  ClawGym RM contains binary
files and thousands of tasks, so this manifest uses the evaluator's
``source_path`` input-file extension.  Files are copied byte-for-byte only when
an attempt is staged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def payload_hash(row: dict[str, Any], length: int = 10) -> str:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


def _source_spec(file_path: str, source: Path) -> dict[str, Any]:
    data = source.read_bytes()
    return {
        "file_path": file_path,
        "source_path": str(source.resolve()),
        "source_size": len(data),
        "source_sha256": hashlib.sha256(data).hexdigest(),
    }


def build_row(task_dir: Path) -> dict[str, Any]:
    entry_path = task_dir / "data_entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    task_id = str(entry.get("task_id") or task_dir.name)
    prompt = str(entry.get("user_query") or "").strip()
    if not prompt:
        raise ValueError(f"empty user_query in {entry_path}")

    input_root = task_dir / "input_files"
    reward_check = task_dir / "reward" / "check.py"
    reward_script = task_dir / "reward" / "reward.sh"
    if not reward_check.is_file() or not reward_script.is_file():
        raise FileNotFoundError(f"missing reward/check.py or reward/reward.sh under {task_dir}")

    input_files = []
    if input_root.is_dir():
        input_files = [
            _source_spec(path.relative_to(input_root).as_posix(), path)
            for path in sorted(input_root.rglob("*"))
            if path.is_file()
        ]
    # internalbench recognizes reward/test.py as the verifier and invokes it
    # with the workspace path.  ClawGym's check.py has the same contract; the
    # original reward.sh simply averages the numeric fields it returns.
    input_files.append(_source_spec("reward/test.py", reward_check))
    return {
        "task_id": task_id,
        "prompt": prompt,
        "hook_code": "python3 reward/test.py /root/.openclaw/workspace",
        "input_files": input_files,
        "rules": None,
        "metadata": {
            **(entry.get("metadata") or {}),
            "task_id": task_id,
            "task_dir": str(task_dir.resolve()),
        },
    }


def _trajectory_has_usage(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                message = event.get("message") if isinstance(event, dict) else None
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict):
                    continue
                fields = ("input", "output", "prompt_tokens", "completion_tokens", "promptTokens", "completionTokens")
                if any(int(usage.get(field) or 0) > 0 for field in fields):
                    return True
    except OSError:
        return False
    return False


def completed_payload_hashes(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not output_dir.exists():
        return completed
    for metadata_path in output_dir.rglob("metadata.json"):
        trajectory_path = metadata_path.with_name("trajectory.jsonl")
        if not trajectory_path.is_file() or not _trajectory_has_usage(trajectory_path):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = str(metadata.get("payload_hash") or "")
        if value:
            completed.add(value)
    return completed


def prepare_manifest(
    dataset_dir: Path,
    output: Path,
    completed_dir: Path | None = None,
    calibration_signature: str = "",
) -> dict[str, int]:
    entry_paths = sorted(dataset_dir.glob("*/data_entry.json"))
    if not entry_paths:
        raise ValueError(f"no */data_entry.json tasks found under {dataset_dir}")
    completed = completed_payload_hashes(completed_dir) if completed_dir else set()
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    skipped = 0
    for entry_path in entry_paths:
        row = build_row(entry_path.parent)
        if calibration_signature:
            row["calibration_signature"] = calibration_signature
        task_id = row["task_id"]
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen_task_ids.add(task_id)
        if payload_hash(row) in completed:
            skipped += 1
            continue
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {"total": len(entry_paths), "pending": len(rows), "completed": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--completed-dir",
        type=Path,
        help="Skip payload hashes with a completed metadata.json and trajectory.jsonl here.",
    )
    parser.add_argument(
        "--calibration-signature",
        default="",
        help="Stable model/pricing signature included in every payload hash.",
    )
    args = parser.parse_args()
    counts = prepare_manifest(
        args.dataset_dir.resolve(),
        args.output.resolve(),
        args.completed_dir,
        args.calibration_signature,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
