#!/usr/bin/env python3
"""Generate visual Sokoban episode skills with an OpenAI-compatible vision API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from idea.analysis import SEEDEpisodeAnalyzer  # noqa: E402


def coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def first_image_path(step: Dict[str, Any]) -> Optional[str]:
    images = step.get("images")
    if not isinstance(images, list) or not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        image = first.get("image")
    else:
        image = first
    return str(image) if image else None


def trajectory_to_seed_steps(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        image_path = first_image_path(step)
        step_record = {
            "step_index": int(step.get("step_idx", len(steps))),
            "observation": str(step.get("observation", "")),
            "observation_prompt": str(
                step.get("observation_prompt")
                or step.get("skill_augmented_observation")
                or step.get("observation", "")
            ),
            "response": str(step.get("model_response") or step.get("raw_action_text") or ""),
            "action_valid": coerce_bool(step.get("action_valid"), default=False),
            "step_reward": step.get("score"),
            "task_description": trajectory.get("task_description", ""),
            "images": step.get("images", []),
        }
        if image_path:
            step_record["observation_image"] = image_path
            step_record["has_observation_image"] = True
        steps.append(step_record)
    return steps


def trajectory_image_paths(trajectory: Dict[str, Any]) -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        image_path = first_image_path(step)
        if image_path:
            images.append({"image": image_path})
    return images


def skill_id_for(trajectory: Dict[str, Any]) -> str:
    baseline_key = str(trajectory.get("baseline_key", "")).strip()
    if baseline_key:
        return baseline_key
    return f"{trajectory.get('task_id', 'sokoban')}:{trajectory.get('rollout_id', 0)}"


def build_candidate_skill_record(
    *,
    trajectory: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    analyzer = SEEDEpisodeAnalyzer(
        backend="openai",
        max_completion_tokens=int(args.skill_max_completion_tokens),
        max_step_skills_per_traj=0,
        skill_mode="episode_only",
        analysis_prompt_version=str(args.analysis_prompt_version),
        include_episode_summary=bool(args.include_episode_summary),
    )
    if args.skill_model:
        analyzer.model = str(args.skill_model)

    steps = trajectory_to_seed_steps(trajectory)
    skill_id = skill_id_for(trajectory)
    image_paths = trajectory_image_paths(trajectory)
    try:
        analysis = analyzer.analyze_episode(
            steps=steps,
            candidate_step_indices=[int(step["step_index"]) for step in steps],
            analysis_mode="teacher_bootstrap",
            episode_success=1.0 if trajectory.get("success") else 0.0,
            task_description=trajectory.get("task_description", ""),
        )
        parse_ok = not bool(analysis.get("analysis_error")) and bool(
            str(analysis.get("episode_skill", "")).strip()
        )
    except Exception as exc:
        analysis = {
            "episode_summary": "",
            "episode_skill": "",
            "step_skills": {},
            "analysis_error": f"{type(exc).__name__}: {exc}",
            "analysis_prompt_version": args.analysis_prompt_version,
            "analysis_backend_requested": "openai",
            "analysis_backend_used": "openai",
            "llm_prompt": None,
            "llm_raw_output": "",
        }
        parse_ok = False

    return {
        "teacher_model": str(args.skill_model or analyzer.model),
        "skill_id": skill_id,
        "task_id": trajectory.get("task_id", ""),
        "task_type": trajectory.get("task_type", "sokoban"),
        "goal_idx": int(trajectory.get("goal_idx", -1)),
        "sample_id": trajectory.get("sample_id"),
        "source_rollout_id": int(trajectory.get("rollout_id", 0)),
        "source_traj_uid": trajectory.get("traj_uid", ""),
        "source_success": bool(trajectory.get("success", False)),
        "source_completed": bool(trajectory.get("completed", False)),
        "source_num_steps": int(trajectory.get("num_steps", len(steps))),
        "source_final_task_score": float(trajectory.get("final_task_score", 0.0)),
        "task_description": trajectory.get("task_description", ""),
        "analysis_prompt_version": analysis.get("analysis_prompt_version", args.analysis_prompt_version),
        "analysis_backend_requested": analysis.get("analysis_backend_requested", "openai"),
        "analysis_backend_used": analysis.get("analysis_backend_used", "openai"),
        "skill_mode": analysis.get("skill_mode", "episode_only"),
        "include_episode_summary": bool(args.include_episode_summary),
        "analysis_prompt": analysis.get("llm_prompt"),
        "analysis_images": image_paths,
        "llm_raw_output": analysis.get("llm_raw_output", ""),
        "llm_raw_outputs": [analysis.get("llm_raw_output", "")] if analysis.get("llm_raw_output") else [],
        "episode_summary": str(analysis.get("episode_summary", "")),
        "episode_skill": str(analysis.get("episode_skill", "")),
        "step_skills": analysis.get("step_skills", {}),
        "parse_ok": bool(parse_ok),
        "analysis_error": analysis.get("analysis_error"),
    }


def generate_candidate_skills(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    if args.regenerate_candidates and path.exists():
        path.unlink()

    existing = read_jsonl(path) if args.resume else []
    existing_ids = {str(record.get("skill_id", "")) for record in existing}
    records = list(existing)
    rollouts = list(baseline_rollouts)[max(0, int(args.candidate_start)) :]
    if args.max_candidates is not None:
        rollouts = rollouts[: max(0, int(args.max_candidates))]

    pending = [trajectory for trajectory in rollouts if skill_id_for(trajectory) not in existing_ids]
    if not pending:
        return records

    worker_count = max(1, min(int(args.skill_gen_workers), len(pending)))
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_trajectory = {
            executor.submit(build_candidate_skill_record, trajectory=trajectory, args=args): trajectory
            for trajectory in pending
        }
        for future in as_completed(future_to_trajectory):
            completed += 1
            record = future.result()
            append_jsonl(path, record)
            records.append(record)
            logging.info(
                "Generated Sokoban candidate skill %d/%d: %s parse_ok=%s error=%s",
                completed,
                len(pending),
                record.get("skill_id"),
                record.get("parse_ok"),
                record.get("analysis_error"),
            )
    return records


def write_metrics(
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
    teacher_model: Optional[str] = None,
) -> None:
    payload = {
        "teacher_model": str(teacher_model or ""),
        "candidate_skills": len(records),
        "parse_ok_skills": sum(1 for record in records if record.get("parse_ok")),
        "parse_error_skills": sum(1 for record in records if not record.get("parse_ok")),
    }
    (output_dir / "candidate_skill_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rollouts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-candidates", action="store_true")
    parser.add_argument("--skill-gen-workers", type=int, default=4)
    parser.add_argument("--skill-model", default=os.environ.get("SKILL_MODEL") or os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--skill-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--skill-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--skill-max-completion-tokens", type=int, default=2048)
    parser.add_argument(
        "--skill-parse-attempts",
        type=int,
        default=int(os.environ.get("SEED_ANALYSIS_PARSE_RETRIES", "2")),
    )
    parser.add_argument(
        "--include-episode-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--analysis-prompt-version", default="seed_visual", choices=("seed_visual",))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    if args.skill_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.skill_api_key)
    if args.skill_base_url:
        os.environ["OPENAI_BASE_URL"] = str(args.skill_base_url)
    if args.skill_model:
        os.environ["OPENAI_MODEL"] = str(args.skill_model)
    os.environ["SEED_ANALYSIS_PARSE_RETRIES"] = str(max(1, int(args.skill_parse_attempts)))

    baseline_path = Path(args.baseline_rollouts).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rollouts = read_jsonl(baseline_path)
    if not baseline_rollouts:
        raise SystemExit(f"No baseline rollouts found in {baseline_path}")

    records = generate_candidate_skills(
        baseline_rollouts=baseline_rollouts,
        output_dir=output_dir,
        args=args,
    )
    write_metrics(records, output_dir, teacher_model=args.skill_model)
    print(
        json.dumps(
            {
                "teacher_model": str(args.skill_model or ""),
                "candidate_skills": len(records),
                "parse_ok_skills": sum(1 for record in records if record.get("parse_ok")),
                "output_path": str(output_dir / "candidate_skills.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
