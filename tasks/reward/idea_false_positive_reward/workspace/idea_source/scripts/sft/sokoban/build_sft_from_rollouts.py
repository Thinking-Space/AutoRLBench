#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def sanitize_path_component(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"


def iter_rollout_entries(rollout_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: (len(p.stem), p.stem)):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                entry.setdefault("_rollout_file", str(path))
                yield entry


def iter_step_entries(rollout_dir: Path) -> Iterable[Dict[str, Any]]:
    for entry in iter_rollout_entries(rollout_dir):
        steps = entry.get("steps")
        if not isinstance(steps, list):
            yield entry
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            step_entry = {
                "_rollout_file": entry.get("_rollout_file"),
                "step": entry.get("source_global_step"),
                "sample_id": entry.get("sample_id"),
                "rollout_id": entry.get("rollout_id"),
                "step_num": step.get("step_idx"),
                "step_id": step.get("source_step_id"),
                "uid": step.get("source_uid") or entry.get("uid"),
                "traj_uid": entry.get("traj_uid"),
                "obs_text": step.get("observation_prompt") or step.get("observation"),
                "obs_text_base": step.get("observation"),
                "input": step.get("observation_prompt") or step.get("observation"),
                "output": step.get("model_response") or step.get("raw_action_text"),
                "score": step.get("score", entry.get("final_task_score")),
                "is_action_valid": step.get("action_valid"),
                "images": step.get("images", []),
            }
            yield step_entry


def boolish(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "valid"}:
        return True
    if text in {"0", "false", "no", "n", "invalid"}:
        return False
    return default


def ensure_image_placeholder(prompt: str) -> str:
    prompt = str(prompt or "").strip()
    if "<image>" in prompt:
        return prompt
    marker = "Your current observation is shown in the image:"
    if marker in prompt:
        return prompt.replace(marker, f"{marker} <image>", 1)
    return f"Your current observation is shown in the image: <image>\n\n{prompt}"


def find_image_path(image_root: Path, entry: Dict[str, Any]) -> Optional[Path]:
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
    return matches[0] if matches else None


def image_path_from_entry(entry: Dict[str, Any]) -> Optional[Path]:
    images = entry.get("images")
    if not isinstance(images, list) or not images:
        return None

    first = images[0]
    if isinstance(first, dict):
        image = first.get("image")
    else:
        image = first
    if not image:
        return None

    path = Path(str(image)).expanduser()
    return path if path.exists() else None


def candidate_images(candidate: Dict[str, Any]) -> List[Dict[str, str]]:
    images = candidate.get("analysis_images")
    if not isinstance(images, list):
        images = candidate.get("images")
    result: List[Dict[str, str]] = []
    if not isinstance(images, list):
        return result
    for item in images:
        if isinstance(item, dict):
            image = item.get("image")
        else:
            image = item
        if image:
            result.append({"image": str(image)})
    return result


def build_records_from_candidate_skills(args: argparse.Namespace, candidate_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped_parse = 0
    skipped_prompt = 0
    skipped_images = 0
    include_summary_arg = getattr(args, "include_episode_summary", None)

    for candidate in iter_rollout_entries(candidate_path.parent):
        if Path(str(candidate.get("_rollout_file", ""))).name != candidate_path.name:
            continue
        if not candidate.get("parse_ok"):
            skipped_parse += 1
            continue
        prompt = candidate.get("analysis_prompt", {})
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        if not messages:
            skipped_prompt += 1
            continue
        prompt_text = str(messages[-1].get("content", ""))
        images = candidate_images(candidate)
        if prompt_text.count("<image>") != len(images):
            skipped_images += 1
            if args.require_images:
                continue

        include_summary = (
            bool(candidate.get("include_episode_summary", True))
            if include_summary_arg is None
            else bool(include_summary_arg)
        )
        response_payload = {"episode_skill": str(candidate.get("episode_skill", ""))}
        if include_summary:
            response_payload = {
                "episode_summary": str(candidate.get("episode_summary", "")),
                **response_payload,
            }
        records.append(
            {
                "prompt": prompt_text,
                "response": json.dumps(response_payload, ensure_ascii=False),
                "teacher_model": candidate.get("teacher_model"),
                "skill_id": candidate.get("skill_id"),
                "task_id": candidate.get("task_id"),
                "task_type": candidate.get("task_type", "sokoban"),
                "goal_idx": candidate.get("goal_idx"),
                "source_rollout_id": candidate.get("source_rollout_id"),
                "source_traj_uid": candidate.get("source_traj_uid"),
                "source_success": candidate.get("source_success"),
                "source_num_steps": candidate.get("source_num_steps"),
                "source_final_task_score": candidate.get("source_final_task_score"),
                "analysis_prompt_version": candidate.get("analysis_prompt_version", "seed_visual"),
                "include_episode_summary": include_summary,
                "parse_ok": bool(candidate.get("parse_ok")),
                "images": images,
            }
        )
        if args.max_records and len(records) >= int(args.max_records):
            break

    print(
        "Built records from candidate skills:",
        len(records),
        "skipped_parse:",
        skipped_parse,
        "skipped_prompt:",
        skipped_prompt,
        "skipped_images:",
        skipped_images,
    )
    return records


def build_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    candidate_path = (
        Path(args.candidate_skills).expanduser().resolve()
        if args.candidate_skills
        else rollout_dir / "candidate_skills.jsonl"
    )
    if candidate_path.exists():
        return build_records_from_candidate_skills(args, candidate_path)

    image_root = Path(args.image_root).expanduser().resolve()
    nested_image_root = image_root / "sokoban_images"
    if nested_image_root.is_dir() and not any(image_root.glob("global_step_*")):
        image_root = nested_image_root
    records: List[Dict[str, Any]] = []
    missing_images = 0
    skipped_invalid = 0
    skipped_score = 0

    for entry in iter_step_entries(rollout_dir):
        if args.valid_actions_only and not boolish(entry.get("is_action_valid"), default=True):
            skipped_invalid += 1
            continue
        if args.min_step_score is not None:
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if score < float(args.min_step_score):
                skipped_score += 1
                continue

        image_path = image_path_from_entry(entry) or find_image_path(image_root=image_root, entry=entry)
        if image_path is None:
            missing_images += 1
            if args.require_images:
                continue

        prompt = entry.get("obs_text") or entry.get("obs_text_base") or entry.get("input") or ""
        response = str(entry.get("output") or "").strip()
        if not prompt or not response:
            continue

        record = {
            "prompt": ensure_image_placeholder(prompt),
            "response": response,
            "source_rollout_file": entry.get("_rollout_file"),
            "source_global_step": entry.get("step"),
            "source_sample_id": entry.get("sample_id"),
            "source_rollout_id": entry.get("rollout_id"),
            "source_step_num": entry.get("step_num"),
            "source_traj_uid": entry.get("traj_uid"),
            "source_score": entry.get("score"),
            "source_action_valid": entry.get("is_action_valid"),
        }
        if image_path is not None:
            record["images"] = [{"image": str(image_path)}]
        records.append(record)
        if args.max_records and len(records) >= int(args.max_records):
            break

    print(
        "Built records:",
        len(records),
        "missing_images:",
        missing_images,
        "skipped_invalid:",
        skipped_invalid,
        "skipped_score:",
        skipped_score,
    )
    return records


def write_splits(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * float(args.val_ratio)))) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    if not train_records and val_records:
        train_records = val_records
        val_records = []

    train_path = output_dir / "sft_episode_skill_train.parquet"
    val_path = output_dir / "sft_episode_skill_val.parquet"
    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    with all_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(val_records or train_records[:1]).to_parquet(val_path)

    summary = {
        "records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records or train_records[:1]),
        "all_jsonl": str(all_jsonl),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (output_dir / "sft_episode_skill_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual Sokoban SFT parquet files from rollout JSONL dumps.")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-skills", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-step-score", type=float, default=None)
    parser.add_argument("--valid-actions-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-episode-summary", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args)
    if not records:
        raise SystemExit("No SFT records were built.")
    write_splits(records, args)


if __name__ == "__main__":
    main()
