#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_DESCRIPTION = "Push all boxes onto targets without trapping boxes."


def sanitize_path_component(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"


def iter_rollout_entries(rollout_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: (len(p.stem), p.stem)):
        if path.name == "baseline_rollouts.jsonl":
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                entry.setdefault("source_rollout_file", str(path))
                yield entry


def extract_existing_image_path(entry: Dict[str, Any]) -> Optional[Path]:
    images = entry.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            image = first.get("image")
        else:
            image = first
        if image:
            return Path(str(image)).expanduser()
    return None


def infer_image_path(image_root: Path, entry: Dict[str, Any]) -> Optional[Path]:
    try:
        global_step = int(entry["step"])
        sample_id = int(entry["sample_id"])
        rollout_id = int(entry["rollout_id"])
        step_num = int(entry["step_num"])
    except (KeyError, TypeError, ValueError):
        return None

    traj_uid = sanitize_path_component(entry.get("traj_uid", "unknown"))
    sequence_name = f"train_sample_{sample_id:06d}_rollout_{rollout_id:03d}_{traj_uid}"
    exact_path = (
        image_root
        / f"global_step_{global_step}"
        / sequence_name
        / f"step_{step_num:03d}.png"
    )
    if exact_path.exists():
        return exact_path

    glob_pattern = (
        f"global_step_{global_step}/"
        f"train_sample_{sample_id:06d}_rollout_{rollout_id:03d}_*/"
        f"step_{step_num:03d}.png"
    )
    matches = sorted(image_root.glob(glob_pattern))
    return matches[0] if matches else exact_path


def add_image(entry: Dict[str, Any], image_root: Path) -> bool:
    image_path = extract_existing_image_path(entry)
    if image_path is None:
        image_path = infer_image_path(image_root=image_root, entry=entry)
    if image_path is None:
        entry["images"] = []
        return False

    entry["images"] = [{"image": str(image_path)}]
    return image_path.exists()


def extract_action_text(text: Any) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return str(text or "").strip()


def entry_sort_key(entry: Dict[str, Any]) -> tuple:
    def as_int(key: str, default: int = 0) -> int:
        try:
            return int(entry.get(key, default))
        except (TypeError, ValueError):
            return default

    return (
        as_int("step"),
        as_int("sample_id"),
        as_int("rollout_id"),
        as_int("step_num"),
    )


def trajectory_key(entry: Dict[str, Any]) -> Tuple[int, int, int, str]:
    def as_int(key: str, default: int = 0) -> int:
        try:
            return int(entry.get(key, default))
        except (TypeError, ValueError):
            return default

    return (
        as_int("step"),
        as_int("sample_id"),
        as_int("rollout_id"),
        str(entry.get("traj_uid", "")),
    )


def build_step_record(entry: Dict[str, Any], next_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    observation = str(entry.get("obs_text_base") or entry.get("obs_text") or "")
    observation_prompt = str(entry.get("obs_text") or entry.get("input") or observation)
    model_response = str(entry.get("output") or "")
    step_record: Dict[str, Any] = {
        "step_idx": int(entry.get("step_num", 0)),
        "observation": observation,
        "observation_prompt": observation_prompt,
        "skill_augmented_observation": observation_prompt,
        "model_response": model_response,
        "raw_action_text": model_response,
        "executed_action": extract_action_text(model_response),
        "action_valid": bool(entry.get("is_action_valid", False)),
        "score": entry.get("score"),
        "done": False,
        "info": {
            "is_action_valid": entry.get("is_action_valid"),
            "score": entry.get("score"),
        },
        "images": entry.get("images", []),
        "source_step_id": entry.get("step_id"),
        "source_uid": entry.get("uid"),
    }
    if next_entry is not None:
        step_record["next_observation"] = str(
            next_entry.get("obs_text_base") or next_entry.get("obs_text") or ""
        )
        step_record["next_observation_prompt"] = str(
            next_entry.get("obs_text") or next_entry.get("input") or step_record["next_observation"]
        )
    else:
        step_record["next_observation"] = ""
        step_record["next_observation_prompt"] = ""
    return step_record


def build_trajectory_record(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = sorted(entries, key=entry_sort_key)
    unique_entries: List[Dict[str, Any]] = []
    seen_steps = set()
    for entry in entries:
        step_key = (entry.get("step_num"), entry.get("step_id"))
        if step_key in seen_steps:
            continue
        seen_steps.add(step_key)
        unique_entries.append(entry)

    first = unique_entries[0]
    last = unique_entries[-1]
    sample_id = int(first.get("sample_id", 0))
    rollout_id = int(first.get("rollout_id", 0))
    final_score = last.get("score", 0.0)
    try:
        final_score_float = float(final_score)
    except (TypeError, ValueError):
        final_score_float = 0.0
    success = final_score_float >= 1.0

    steps = [
        build_step_record(entry, unique_entries[index + 1] if index + 1 < len(unique_entries) else None)
        for index, entry in enumerate(unique_entries)
    ]
    if steps:
        steps[-1]["done"] = bool(success)

    source_files = sorted({str(entry.get("source_rollout_file", "")) for entry in unique_entries if entry.get("source_rollout_file")})
    return {
        "task_id": f"sokoban_{sample_id:06d}",
        "task_type": "sokoban",
        "goal_idx": sample_id,
        "sample_id": sample_id,
        "source_skill_id": None,
        "rollout_id": rollout_id,
        "seed": None,
        "history_length": None,
        "episode_skill": "",
        "task_description": TASK_DESCRIPTION,
        "initial_info": {},
        "traj_uid": str(first.get("traj_uid", "")),
        "uid": first.get("uid"),
        "source_global_step": first.get("step"),
        "source_rollout_file": source_files[0] if len(source_files) == 1 else source_files,
        "steps": steps,
        "success": bool(success),
        "completed": bool(success),
        "num_steps": len(steps),
        "final_reward": final_score_float,
        "final_task_score": final_score_float,
        "baseline_key": f"sokoban_{sample_id:06d}:{rollout_id}",
    }


def group_trajectories(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(trajectory_key(entry), []).append(entry)
    return [
        build_trajectory_record(grouped[key])
        for key in sorted(grouped)
        if grouped[key]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sokoban raw rollout dumps into baseline_rollouts.jsonl.")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--require-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    nested_image_root = image_root / "sokoban_images"
    if nested_image_root.is_dir() and not any(image_root.glob("global_step_*")):
        image_root = nested_image_root

    step_entries: List[Dict[str, Any]] = []
    missing_images = 0
    for entry in iter_rollout_entries(rollout_dir):
        if not add_image(entry, image_root=image_root):
            missing_images += 1
        step_entries.append(entry)

    if not step_entries:
        raise SystemExit(f"No rollout JSONL entries found in {rollout_dir}")
    if args.require_images and missing_images:
        raise SystemExit(
            f"Missing {missing_images} image(s) while exporting {len(step_entries)} rollout step entries. "
            f"rollout_dir={rollout_dir} image_root={image_root}"
        )
    trajectories = group_trajectories(step_entries)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    summary = {
        "records": len(trajectories),
        "step_records": len(step_entries),
        "missing_images": missing_images,
        "rollout_dir": str(rollout_dir),
        "image_root": str(image_root),
        "output_path": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
