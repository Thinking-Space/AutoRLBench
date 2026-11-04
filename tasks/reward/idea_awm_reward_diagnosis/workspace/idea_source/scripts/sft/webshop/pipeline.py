#!/usr/bin/env python3
"""Generate episode-skill SFT data from WebShop policy rollouts.

This is the WebShop counterpart of the ALFWorld pipeline: sample train
goals, collect baseline rollouts with a local/OpenAI-compatible policy model,
ask an LLM to produce episode-level skills with the current SEED analyzer
prompt, and export all parseable skills directly as SFT data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from examples.prompt_agent.local_vllm_alfworld import (  # noqa: E402
    AttrDict,
    LocalOpenAIAgent,
    collect_actions_concurrently,
    json_safe,
    load_env_file,
    resolve_extra_body,
)
from examples.prompt_agent.local_vllm_webshop import has_action_tag  # noqa: E402
from idea.prompting import build_augmented_observation_text  # noqa: E402
from scripts.sft._common.pipeline import (  # noqa: E402
    ChatEndpoint,
    OpenAITextClient,
    append_jsonl,
    coerce_bool,
    log_stage,
    normalize_messages,
    read_jsonl,
    resolve_endpoint,
    setup_logging,
    update_progress,
    write_json,
)


OUTPUT_FILES = [
    "sampled_tasks.jsonl",
    "baseline_rollouts.jsonl",
    "candidate_skills.jsonl",
    "sft_episode_skill_all.jsonl",
    "sft_episode_skill_train.parquet",
    "sft_episode_skill_val.parquet",
    "sft_filter_metrics.json",
    "metrics.json",
    "progress.json",
    "run_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs/webshop_episode_skill_pipeline")
    parser.add_argument("--num-tasks", type=int, default=180)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=16,
        help="Number of different WebShop goals to rollout in the same wave.",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument(
        "--baseline-history-length",
        type=int,
        default=None,
        help=(
            "Prompt history length used only while collecting baseline rollouts. "
            "Defaults to --history-length for backward compatibility."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--request-workers", type=int, default=128)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--baseline-rollouts",
        default=None,
        help="Reuse an existing baseline_rollouts.jsonl instead of collecting policy rollouts.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--regenerate-candidates",
        action="store_true",
        help="Delete existing candidate/SFT outputs before generating candidates.",
    )
    parser.add_argument(
        "--stop-after-baseline-rollouts",
        action="store_true",
        help="Exit successfully after baseline rollouts are complete.",
    )
    parser.add_argument(
        "--stop-after-skill-generation",
        action="store_true",
        help="Exit successfully after candidate skill API generation is complete.",
    )
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--webshop-data-dir", default=None)
    parser.add_argument("--webshop-human-goals", type=int, default=0)
    parser.add_argument("--webshop-use-small", action="store_true")
    parser.add_argument("--webshop-train-start", type=int, default=500)
    parser.add_argument("--webshop-train-end", type=int, default=None)
    parser.add_argument("--num-cpus-per-env-worker", type=float, default=0.05)
    parser.add_argument("--startup-wait-seconds", type=float, default=1.0)

    parser.add_argument("--policy-base-url", default=None)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--policy-model", default=None)
    parser.add_argument("--policy-temperature", type=float, default=0.4)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=512)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--policy-retries", type=int, default=2)
    parser.add_argument("--policy-retry-delay", type=float, default=1.0)
    parser.add_argument("--policy-extra-body-json", default=None)
    parser.add_argument("--fallback-action", default="click[back to search]")

    parser.add_argument("--skill-base-url", default=None)
    parser.add_argument("--skill-api-key", default=None)
    parser.add_argument("--skill-model", default=None)
    parser.add_argument("--skill-temperature", type=float, default=0.0)
    parser.add_argument("--skill-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--skill-timeout", type=float, default=120.0)
    parser.add_argument("--skill-retries", type=int, default=5)
    parser.add_argument("--skill-retry-delay", type=float, default=1.0)
    parser.add_argument("--skill-extra-body-json", default=None)
    parser.add_argument("--skill-gen-workers", type=int, default=128)
    parser.add_argument(
        "--skill-parse-attempts",
        type=int,
        default=2,
        help="Maximum API calls per skill sample when the previous response cannot be parsed as valid JSON.",
    )
    parser.add_argument(
        "--include-episode-summary",
        type=lambda value: coerce_bool(value, default=True),
        default=True,
        help="Whether skill-analysis prompts and SFT responses include episode_summary.",
    )
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--sft-min-source-score",
        type=float,
        default=0.0,
        help=(
            "Keep parseable SFT examples whose source final_task_score is at least this value. "
            "Successful examples can still be kept by --sft-include-success."
        ),
    )
    parser.add_argument(
        "--sft-include-success",
        type=lambda value: coerce_bool(value, default=True),
        default=True,
        help="Whether to always keep parseable examples from successful source rollouts.",
    )
    parser.add_argument(
        "--sft-max-zero-score-failures",
        type=int,
        default=None,
        help=(
            "Optional cap for parseable failed examples with source_final_task_score <= 0. "
            "Use 0 to drop all zero-score failures."
        ),
    )
    parser.add_argument(
        "--sft-max-records",
        type=int,
        default=None,
        help="Optional cap on exported SFT records after sorting by success, score, and length.",
    )
    return parser.parse_args()


def prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
    regenerate_candidates: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if regenerate_candidates:
        for name in (
            "candidate_skills.jsonl",
            "sft_episode_skill_all.jsonl",
            "sft_episode_skill_train.parquet",
            "sft_episode_skill_val.parquet",
            "sft_filter_metrics.json",
            "metrics.json",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()
        return
    if existing and not resume:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. "
            "Use --resume, --overwrite, or --regenerate-candidates."
        )


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def resolve_webshop_data_paths(args: argparse.Namespace) -> Dict[str, str]:
    default_data_dir = PROJECT_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    data_dir = Path(
        args.webshop_data_dir
        or os.environ.get("WEBSHOP_DATA")
        or os.environ.get("WEBSHOP_DATA_DIR")
        or default_data_dir
    )
    if args.webshop_use_small:
        file_name = "items_shuffle_1000.json"
        attr_name = "items_ins_v2_1000.json"
    else:
        file_name = "items_shuffle.json"
        attr_name = "items_ins_v2.json"

    file_path = data_dir / file_name
    attr_path = data_dir / attr_name
    missing = [str(path) for path in (file_path, attr_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WebShop data files not found: "
            + ", ".join(missing)
            + ". Set WEBSHOP_DATA/WEBSHOP_DATA_DIR or pass --webshop-data-dir."
        )
    return {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": int(args.webshop_human_goals),
        "file_path": str(file_path),
        "attr_path": str(attr_path),
    }


def get_webshop_goals(args: argparse.Namespace) -> List[str]:
    import ray
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    env_kwargs = resolve_webshop_data_paths(args)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    worker_cls = ray.remote(num_cpus=max(float(args.num_cpus_per_env_worker), 0.01))(WebshopWorker)
    worker = worker_cls.remote(int(args.seed), dict(env_kwargs))
    try:
        return list(ray.get(worker.get_goals.remote()))
    finally:
        try:
            ray.get(worker.close.remote())
        except Exception:
            pass
        ray.kill(worker)


def sample_tasks(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    sampled_path = output_dir / "sampled_tasks.jsonl"
    existing = read_jsonl(sampled_path)
    if existing and args.resume:
        return existing

    if args.baseline_rollouts:
        baseline_records = read_jsonl(Path(args.baseline_rollouts))
        if not baseline_records:
            raise ValueError(f"No baseline rollouts found in {args.baseline_rollouts}")
        tasks_by_id: Dict[str, Dict[str, Any]] = {}
        for record in baseline_records:
            task_id = str(record.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("External baseline rollout is missing task_id")
            tasks_by_id.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_type": record.get("task_type", "webshop"),
                    "goal_idx": int(record.get("goal_idx", -1)),
                    "sample_index": len(tasks_by_id),
                    "split": "train",
                    "task_description": str(record.get("task_description", "")),
                },
            )
        sampled = list(tasks_by_id.values())
        if sampled_path.exists():
            sampled_path.unlink()
        for record in sampled:
            append_jsonl(sampled_path, record)
        logging.info(
            "Derived %d WebShop tasks from external baseline rollouts.",
            len(sampled),
        )
        return sampled

    goals = get_webshop_goals(args)
    start = max(0, int(args.webshop_train_start))
    end = len(goals) if args.webshop_train_end is None else min(len(goals), int(args.webshop_train_end))
    if end <= start:
        raise ValueError(
            f"Invalid WebShop train goal range [{start}, {end}) for {len(goals)} available goals."
        )
    pool = list(range(start, end))
    if len(pool) < int(args.num_tasks):
        raise ValueError(
            f"Need {args.num_tasks} WebShop goals, found {len(pool)} in range [{start}, {end})."
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sampled: List[Dict[str, Any]] = []
    for sample_idx, goal_idx in enumerate(pool[: int(args.num_tasks)]):
        sampled.append(
            {
                "task_id": f"webshop_{goal_idx:06d}",
                "task_type": "webshop",
                "goal_idx": int(goal_idx),
                "sample_index": int(sample_idx),
                "split": "train",
                "task_description": str(goals[goal_idx]),
            }
        )

    if args.max_tasks is not None:
        sampled = sampled[: max(0, int(args.max_tasks))]

    if sampled_path.exists():
        sampled_path.unlink()
    for record in sampled:
        append_jsonl(sampled_path, record)
    logging.info("Sampled %d WebShop train goals.", len(sampled))
    return sampled


class FixedWebshopSessionBatchEnv:
    """A Ray-backed WebShop env batch bound to explicit goal/session indices."""

    def __init__(
        self,
        *,
        session_indices: Sequence[int],
        seed: int,
        env_kwargs: Dict[str, Any],
        num_cpus_per_worker: float,
    ) -> None:
        import ray
        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        self.session_indices = [int(idx) for idx in session_indices]
        if not self.session_indices:
            raise ValueError("FixedWebshopSessionBatchEnv requires at least one session index.")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        self._ray = ray
        worker_cls = ray.remote(num_cpus=max(float(num_cpus_per_worker), 0.01))(WebshopWorker)
        self._workers = [
            worker_cls.remote(int(seed) + worker_idx, dict(env_kwargs))
            for worker_idx in range(len(self.session_indices))
        ]
        self._closed = False

    def reset(self):
        futures = [
            worker.reset.remote(session_idx)
            for worker, session_idx in zip(self._workers, self.session_indices)
        ]
        results = self._ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: Sequence[str]):
        futures = [worker.step.remote(action) for worker, action in zip(self._workers, actions)]
        results = self._ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if self._closed:
            return
        close_futures = []
        for worker in self._workers:
            try:
                close_futures.append(worker.close.remote())
            except Exception:
                pass
        if close_futures:
            try:
                self._ray.get(close_futures)
            except Exception:
                pass
        for worker in self._workers:
            try:
                self._ray.kill(worker)
            except Exception:
                pass
        self._closed = True


def build_manager(
    *,
    session_indices: Sequence[int],
    seed: int,
    args: argparse.Namespace,
) -> Any:
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection

    envs = FixedWebshopSessionBatchEnv(
        session_indices=session_indices,
        seed=seed,
        env_kwargs=resolve_webshop_data_paths(args),
        num_cpus_per_worker=args.num_cpus_per_env_worker,
    )
    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "webshop",
                    "history_length": baseline_history_length(args),
                    "use_skills_only_memory": False,
                }
            )
        }
    )
    time.sleep(max(float(args.startup_wait_seconds), len(session_indices) * 0.02))
    return WebshopEnvironmentManager(envs, webshop_projection, config)


def baseline_history_length(args: argparse.Namespace) -> int:
    value = args.baseline_history_length
    if value is None:
        value = args.history_length
    return int(value)


def make_policy_agent(endpoint: ChatEndpoint, fallback_action: str) -> LocalOpenAIAgent:
    return LocalOpenAIAgent(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        model_name=endpoint.model,
        temperature=endpoint.temperature,
        max_completion_tokens=endpoint.max_completion_tokens,
        timeout=endpoint.timeout,
        retries=endpoint.retries,
        retry_delay=endpoint.retry_delay,
        fallback_action=fallback_action,
        extra_body=endpoint.extra_body,
    )


def extract_action_text(model_response: str) -> str:
    text = str(model_response or "")
    lowered = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return text[-50:].strip().lower()
    return text[start_idx + len(start_tag) : end_idx].strip().lower()


def run_rollout_specs(
    *,
    specs: Sequence[Dict[str, Any]],
    seed: int,
    args: argparse.Namespace,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    if not specs:
        return []

    batch_size = len(specs)
    manager = build_manager(
        session_indices=[spec["task"]["goal_idx"] for spec in specs],
        seed=seed,
        args=args,
    )
    agent = make_policy_agent(policy_endpoint, fallback_action=args.fallback_action)
    request_workers = max(1, min(int(args.request_workers), batch_size))
    trajectories: List[Dict[str, Any]] = []
    try:
        obs, infos = manager.reset({})
        env_dones = [False] * batch_size
        success_flags = np.zeros(batch_size, dtype=bool)
        final_task_scores = np.zeros(batch_size, dtype=float)
        task_descriptions = list(getattr(manager, "tasks", [""] * batch_size))
        trajectories = [
            {
                "task_id": spec["task"]["task_id"],
                "task_type": spec["task"].get("task_type", "webshop"),
                "goal_idx": int(spec["task"]["goal_idx"]),
                "source_skill_id": spec.get("source_skill_id"),
                "rollout_id": int(spec.get("rollout_id", env_idx)),
                "seed": int(spec.get("seed", seed + env_idx)),
                "history_length": baseline_history_length(args),
                "episode_skill": spec.get("episode_skill", ""),
                "task_description": (
                    task_descriptions[env_idx]
                    if env_idx < len(task_descriptions) and task_descriptions[env_idx]
                    else spec["task"].get("task_description", "")
                ),
                "initial_info": json_safe(infos[env_idx]),
                "steps": [],
            }
            for env_idx, spec in enumerate(specs)
        ]

        for step_idx in range(int(args.max_steps)):
            current_prompts = [str(item) for item in obs.get("text", [])]
            current_observations = [str(item) for item in obs.get("anchor", current_prompts)]
            agent_observations = []
            for env_idx, observation_prompt in enumerate(current_prompts):
                env_skill = str(specs[env_idx].get("episode_skill", "")).strip()
                if env_skill:
                    agent_observations.append(
                        build_augmented_observation_text(
                            observation=observation_prompt,
                            episode_skill=env_skill,
                        )
                    )
                else:
                    agent_observations.append(observation_prompt)

            actions, model_responses, _format_flags, _error_flags = collect_actions_concurrently(
                agent=agent,
                observations=agent_observations,
                env_dones=env_dones,
                done_action="<think>The episode is done.</think><action>click[back to search]</action>",
                request_workers=request_workers,
                log_responses=False,
                format_checker=has_action_tag,
            )

            next_obs, rewards, dones, infos = manager.step(list(actions))
            next_prompts = [str(item) for item in next_obs.get("text", [])]
            next_observations = [str(item) for item in next_obs.get("anchor", next_prompts)]

            for env_idx in range(batch_size):
                if env_dones[env_idx]:
                    continue
                action_valid = coerce_bool(infos[env_idx].get("is_action_valid"), default=False)
                final_task_scores[env_idx] = float(
                    infos[env_idx].get("task_score", final_task_scores[env_idx])
                )
                trajectories[env_idx]["steps"].append(
                    {
                        "step_idx": int(step_idx),
                        "observation": current_observations[env_idx],
                        "observation_prompt": current_prompts[env_idx],
                        "skill_augmented_observation": agent_observations[env_idx],
                        "model_response": model_responses[env_idx],
                        "raw_action_text": actions[env_idx],
                        "executed_action": extract_action_text(actions[env_idx]),
                        "action_valid": action_valid,
                        "reward": json_safe(rewards[env_idx]),
                        "done": bool(dones[env_idx]),
                        "info": json_safe(infos[env_idx]),
                        "next_observation": next_observations[env_idx],
                        "next_observation_prompt": next_prompts[env_idx],
                    }
                )

            obs = next_obs
            for env_idx in range(batch_size):
                if env_dones[env_idx]:
                    continue
                if dones[env_idx]:
                    env_dones[env_idx] = True
                    success_flags[env_idx] = bool(infos[env_idx].get("won", False))
            if all(env_dones):
                break

        for env_idx, trajectory in enumerate(trajectories):
            trajectory["success"] = bool(success_flags[env_idx])
            trajectory["completed"] = bool(env_dones[env_idx])
            trajectory["num_steps"] = len(trajectory["steps"])
            trajectory["final_reward"] = (
                trajectory["steps"][-1]["reward"] if trajectory["steps"] else 0.0
            )
            trajectory["final_task_score"] = float(final_task_scores[env_idx])
    finally:
        manager.close()
    return trajectories


def collect_baseline_rollouts(
    *,
    tasks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "baseline_rollouts.jsonl"
    if args.baseline_rollouts:
        source_path = Path(args.baseline_rollouts).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Baseline rollouts not found: {source_path}")
        if source_path != path.resolve():
            if path.exists():
                path.unlink()
            shutil.copy2(source_path, path)
        records = read_jsonl(path)
        task_ids = {str(record.get("task_id", "")) for record in records}
        if len(task_ids) != len(tasks):
            raise ValueError(
                "External baseline/task mismatch: "
                f"{len(task_ids)} baseline tasks versus {len(tasks)} sampled tasks"
            )
        logging.info(
            "Reused %d baseline rollouts from %s.",
            len(records),
            source_path,
        )
        log_stage(
            output_dir,
            "baseline_rollout",
            "complete",
            completed_rollouts=len(records),
            expected_rollouts=len(records),
            reused_from=str(source_path),
        )
        return records

    existing = read_jsonl(path) if args.resume else []
    existing_keys = {(record["task_id"], int(record["rollout_id"])) for record in existing}
    records = list(existing)
    expected_total = len(tasks) * int(args.rollouts_per_task)
    log_stage(
        output_dir,
        "baseline_rollout",
        "running",
        existing_rollouts=len(records),
        expected_rollouts=expected_total,
        task_batch_size=int(args.task_batch_size),
        rollouts_per_task=int(args.rollouts_per_task),
        history_length=baseline_history_length(args),
    )

    pending_specs: List[Dict[str, Any]] = []
    for task_idx, task in enumerate(tasks):
        for rollout_id in range(int(args.rollouts_per_task)):
            if (task["task_id"], rollout_id) in existing_keys:
                continue
            pending_specs.append(
                {
                    "task": task,
                    "task_index": int(task_idx),
                    "rollout_id": int(rollout_id),
                    "seed": int(args.seed) + task_idx * 1000 + rollout_id,
                }
            )

    if not pending_specs:
        log_stage(
            output_dir,
            "baseline_rollout",
            "complete",
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
            history_length=baseline_history_length(args),
        )
        return records

    env_batch_size = max(1, int(args.task_batch_size)) * max(1, int(args.rollouts_per_task))
    total_waves = (len(pending_specs) + env_batch_size - 1) // env_batch_size
    for wave_idx, spec_chunk in enumerate(chunked(pending_specs, env_batch_size)):
        task_ids = sorted({spec["task"]["task_id"] for spec in spec_chunk})
        logging.info(
            "Baseline wave %d/%d: %d envs across %d WebShop goal(s).",
            wave_idx + 1,
            total_waves,
            len(spec_chunk),
            len(task_ids),
        )
        update_progress(
            output_dir,
            stage="baseline_rollout",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            envs_in_wave=len(spec_chunk),
            tasks_in_wave=len(task_ids),
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
            history_length=baseline_history_length(args),
        )
        trajectories = run_rollout_specs(
            specs=spec_chunk,
            seed=int(args.seed) + wave_idx * 100_000,
            args=args,
            policy_endpoint=policy_endpoint,
        )
        for trajectory in trajectories:
            trajectory["baseline_key"] = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            append_jsonl(path, trajectory)
            records.append(trajectory)
        update_progress(
            output_dir,
            stage="baseline_rollout",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
            history_length=baseline_history_length(args),
        )

    log_stage(
        output_dir,
        "baseline_rollout",
        "complete",
        completed_rollouts=len(records),
        expected_rollouts=expected_total,
        history_length=baseline_history_length(args),
    )
    return records


def trajectory_to_seed_steps(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        step_info = step.get("info", {})
        action_valid = step.get("action_valid")
        if action_valid is None and isinstance(step_info, dict):
            action_valid = step_info.get("is_action_valid")
        steps.append(
            {
                "step_index": int(step.get("step_idx", len(steps))),
                "observation": str(step.get("observation", "")),
                "observation_prompt": str(
                    step.get("observation_prompt")
                    or step.get("skill_augmented_observation")
                    or step.get("observation", "")
                ),
                "response": str(step.get("model_response", "")),
                "action_valid": coerce_bool(action_valid, default=False),
                "task_description": trajectory.get("task_description", ""),
            }
        )
    return steps


def build_candidate_skill_record(
    *,
    trajectory: Dict[str, Any],
    skill_endpoint: ChatEndpoint,
    include_episode_summary: bool = True,
    skill_parse_attempts: int = 2,
) -> Dict[str, Any]:
    from idea.analysis import SEEDEpisodeAnalyzer

    analyzer = SEEDEpisodeAnalyzer(
        backend="openai",
        max_completion_tokens=skill_endpoint.max_completion_tokens,
        max_step_skills_per_traj=0,
        skill_mode="episode_only",
        include_episode_summary=include_episode_summary,
    )
    skill_client = OpenAITextClient(skill_endpoint)
    skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
    steps = trajectory_to_seed_steps(trajectory)
    prompt = analyzer._build_episode_analysis_prompt(
        steps=steps,
        candidate_step_indices=[step["step_index"] for step in steps],
        analysis_mode="teacher_bootstrap",
        episode_success=1.0 if trajectory.get("success") else 0.0,
        task_description=trajectory.get("task_description", ""),
    )
    parse_ok = False
    parsed: Dict[str, Any] = {"episode_summary": "", "episode_skill": ""}
    raw_output = ""
    api_error = None
    parse_error = None
    raw_outputs: List[str] = []
    prompt_for_attempt = prompt
    attempts_used = 0
    for attempt_idx in range(max(1, int(skill_parse_attempts))):
        attempts_used = attempt_idx + 1
        raw_output, api_error = skill_client.complete(normalize_messages(prompt_for_attempt))
        raw_outputs.append(raw_output)
        if api_error:
            parse_error = None
            break
        try:
            parsed = analyzer._parse_analysis_response(raw_output)
            if not str(parsed.get("episode_skill", "")).strip():
                raise ValueError("SEED analyzer response missing required field: episode_skill")
            parse_ok = True
            parse_error = None
            break
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            if attempt_idx + 1 >= max(1, int(skill_parse_attempts)):
                break
            prompt_for_attempt = analyzer._build_json_retry_prompt(
                original_prompt=prompt,
                invalid_response=raw_output,
                error=exc,
            )

    return {
        "skill_id": skill_id,
        "task_id": trajectory["task_id"],
        "task_type": trajectory.get("task_type", "webshop"),
        "goal_idx": int(trajectory.get("goal_idx", -1)),
        "source_rollout_id": int(trajectory["rollout_id"]),
        "source_success": bool(trajectory.get("success", False)),
        "source_num_steps": int(trajectory.get("num_steps", 0)),
        "source_final_task_score": float(trajectory.get("final_task_score", 0.0)),
        "task_description": trajectory.get("task_description", ""),
        "analysis_prompt": prompt,
        "llm_raw_output": raw_output,
        "llm_raw_outputs": raw_outputs,
        "episode_summary": str(parsed.get("episode_summary", "")),
        "episode_skill": str(parsed.get("episode_skill", "")),
        "parse_ok": parse_ok,
        "parse_attempts": attempts_used,
        "max_parse_attempts": max(1, int(skill_parse_attempts)),
        "analysis_error": api_error or parse_error,
    }


def generate_candidate_skills(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    output_dir: Path,
    skill_endpoint: ChatEndpoint,
    max_candidates: Optional[int],
    max_workers: int,
    resume: bool,
    include_episode_summary: bool,
    skill_parse_attempts: int,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    existing = read_jsonl(path) if resume else []
    existing_ids = {record["skill_id"] for record in existing}
    records = list(existing)
    rollouts = list(baseline_rollouts)
    if max_candidates is not None:
        rollouts = rollouts[: max(0, int(max_candidates))]

    pending = [
        trajectory
        for trajectory in rollouts
        if f"{trajectory['task_id']}:{trajectory['rollout_id']}" not in existing_ids
    ]
    expected_total = len(rollouts)
    log_stage(
        output_dir,
        "skill_generation",
        "running",
        existing_skills=len(existing),
        pending_skills=len(pending),
        expected_skills=expected_total,
        skill_gen_workers=int(max_workers),
        skill_parse_attempts=max(1, int(skill_parse_attempts)),
    )
    if not pending:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(records),
            expected_skills=expected_total,
            parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
        )
        return records

    worker_count = max(1, min(int(max_workers), len(pending)))
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_trajectory = {
            executor.submit(
                build_candidate_skill_record,
                trajectory=trajectory,
                skill_endpoint=skill_endpoint,
                include_episode_summary=include_episode_summary,
                skill_parse_attempts=skill_parse_attempts,
            ): trajectory
            for trajectory in pending
        }
        for future in as_completed(future_to_trajectory):
            trajectory = future_to_trajectory[future]
            skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            completed += 1
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "skill_id": skill_id,
                    "task_id": trajectory["task_id"],
                    "task_type": trajectory.get("task_type", "webshop"),
                    "goal_idx": int(trajectory.get("goal_idx", -1)),
                    "source_rollout_id": int(trajectory["rollout_id"]),
                    "source_success": bool(trajectory.get("success", False)),
                    "source_num_steps": int(trajectory.get("num_steps", 0)),
                    "source_final_task_score": float(trajectory.get("final_task_score", 0.0)),
                    "task_description": trajectory.get("task_description", ""),
                    "analysis_prompt": None,
                    "llm_raw_output": "",
                    "llm_raw_outputs": [],
                    "episode_summary": "",
                    "episode_skill": "",
                    "parse_ok": False,
                    "parse_attempts": 0,
                    "max_parse_attempts": max(1, int(skill_parse_attempts)),
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(path, record)
            records.append(record)
            logging.info(
                "Generated candidate skill %d/%d: %s parse_ok=%s.",
                completed,
                len(pending),
                skill_id,
                record.get("parse_ok"),
            )
            update_progress(
                output_dir,
                stage="skill_generation",
                status="running",
                completed_in_current_run=completed,
                pending_in_current_run=len(pending),
                completed_skills=len(records),
                expected_skills=expected_total,
                last_skill_id=skill_id,
                last_parse_ok=record.get("parse_ok"),
            )

    log_stage(
        output_dir,
        "skill_generation",
        "complete",
        completed_skills=len(records),
        expected_skills=expected_total,
        parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
    )
    return records


def build_sft_exports_from_candidates(
    *,
    candidate_skills: Sequence[Dict[str, Any]],
    output_dir: Path,
    sft_val_ratio: float,
    seed: int,
    sft_min_source_score: float,
    sft_include_success: bool,
    sft_max_zero_score_failures: Optional[int],
    sft_max_records: Optional[int],
    include_episode_summary: bool,
) -> List[Dict[str, Any]]:
    sft_records: List[Dict[str, Any]] = []
    filter_counts: Counter[str] = Counter()
    source_score_bins_before: Counter[str] = Counter()
    source_score_bins_after: Counter[str] = Counter()
    zero_score_failures_kept = 0

    def score_bucket(score: float) -> str:
        if score >= 1.0:
            return ">=1.0"
        if score >= 0.8:
            return "0.8-1.0"
        if score >= 0.6:
            return "0.6-0.8"
        if score >= 0.4:
            return "0.4-0.6"
        if score > 0.0:
            return "0.0-0.4"
        return "0.0"

    for candidate in candidate_skills:
        if not candidate.get("parse_ok"):
            filter_counts["drop_parse_error"] += 1
            continue
        prompt = candidate.get("analysis_prompt", {})
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        if not messages:
            filter_counts["drop_missing_prompt"] += 1
            continue

        source_success = bool(candidate.get("source_success", False))
        source_score = float(candidate.get("source_final_task_score", 0.0))
        source_num_steps = int(candidate.get("source_num_steps", 0))
        source_bucket = score_bucket(source_score)
        source_score_bins_before[source_bucket] += 1

        keep_reason = ""
        if source_success and sft_include_success:
            keep_reason = "success"
        elif source_score >= float(sft_min_source_score):
            keep_reason = "score_threshold"
        elif source_score <= 0.0 and sft_max_zero_score_failures is not None:
            if zero_score_failures_kept < max(0, int(sft_max_zero_score_failures)):
                zero_score_failures_kept += 1
                keep_reason = "zero_score_failure_cap"

        if not keep_reason:
            filter_counts["drop_low_score"] += 1
            continue

        response_payload = {"episode_skill": candidate.get("episode_skill", "")}
        if include_episode_summary:
            response_payload = {
                "episode_summary": candidate.get("episode_summary", ""),
                **response_payload,
            }
        source_score_bins_after[source_bucket] += 1
        filter_counts[f"keep_{keep_reason}"] += 1
        record = (
            {
                "prompt": str(messages[-1].get("content", "")),
                "response": json.dumps(response_payload, ensure_ascii=False),
                "skill_id": candidate["skill_id"],
                "task_id": candidate["task_id"],
                "task_type": candidate.get("task_type", "webshop"),
                "goal_idx": int(candidate.get("goal_idx", -1)),
                "source_success": source_success,
                "source_num_steps": source_num_steps,
                "source_final_task_score": source_score,
                "source_score_bucket": source_bucket,
                "sft_filter_reason": keep_reason,
                "include_episode_summary": bool(include_episode_summary),
                "parse_ok": bool(candidate.get("parse_ok")),
            }
        )
        sft_records.append(record)

    if sft_max_records is not None and int(sft_max_records) > 0:
        before_cap = len(sft_records)
        sft_records = sorted(
            sft_records,
            key=lambda record: (
                not bool(record.get("source_success", False)),
                -float(record.get("source_final_task_score", 0.0)),
                int(record.get("source_num_steps", 10**9)),
                str(record.get("skill_id", "")),
            ),
        )[: int(sft_max_records)]
        filter_counts["drop_max_records_cap"] += max(0, before_cap - len(sft_records))

    filter_metrics = {
        "total_candidates": len(candidate_skills),
        "exported_sft_records": len(sft_records),
        "sft_min_source_score": float(sft_min_source_score),
        "sft_include_success": bool(sft_include_success),
        "sft_max_zero_score_failures": sft_max_zero_score_failures,
        "sft_max_records": sft_max_records,
        "include_episode_summary": bool(include_episode_summary),
        "filter_counts": dict(filter_counts),
        "source_score_bins_before": dict(source_score_bins_before),
        "source_score_bins_after": dict(Counter(record.get("source_score_bucket", "unknown") for record in sft_records)),
        "source_success_counts_after": dict(
            Counter(str(bool(record.get("source_success", False))) for record in sft_records)
        ),
    }
    write_json(output_dir / "sft_filter_metrics.json", filter_metrics)

    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in sft_records:
        append_jsonl(all_jsonl, record)

    if not sft_records:
        logging.warning("No parseable candidate skills; SFT parquet export skipped.")
        return sft_records

    shuffled = list(sft_records)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * sft_val_ratio))
    if len(shuffled) > 1:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    try:
        import pandas as pd

        pd.DataFrame(train_records).to_parquet(output_dir / "sft_episode_skill_train.parquet")
        pd.DataFrame(val_records).to_parquet(output_dir / "sft_episode_skill_val.parquet")
    except Exception as exc:
        logging.warning("Could not write parquet SFT exports: %s", exc)
    return sft_records


def write_metrics(
    *,
    tasks: Sequence[Dict[str, Any]],
    baseline_rollouts: Sequence[Dict[str, Any]],
    candidate_skills: Sequence[Dict[str, Any]],
    sft_records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    final_scores = [float(record.get("final_task_score", 0.0)) for record in baseline_rollouts]
    payload = {
        "version": "webshop_no_skill_validation",
        "sampled_tasks": len(tasks),
        "baseline_rollouts": len(baseline_rollouts),
        "candidate_skills": len(candidate_skills),
        "parse_ok_skills": sum(1 for record in candidate_skills if record.get("parse_ok")),
        "parse_error_skills": sum(1 for record in candidate_skills if not record.get("parse_ok")),
        "sft_records": len(sft_records),
        "candidate_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in candidate_skills)),
        "sft_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in sft_records)),
        "source_success_counts": dict(Counter(str(bool(record.get("source_success", False))) for record in candidate_skills)),
        "sft_source_success_counts": dict(Counter(str(bool(record.get("source_success", False))) for record in sft_records)),
        "sft_source_score_bins": dict(Counter(record.get("source_score_bucket", "unknown") for record in sft_records)),
        "sft_filter_reasons": dict(Counter(record.get("sft_filter_reason", "unknown") for record in sft_records)),
        "baseline_success_rate": (
            float(np.mean([bool(record.get("success", False)) for record in baseline_rollouts]))
            if baseline_rollouts
            else 0.0
        ),
        "baseline_final_task_score_mean": float(np.mean(final_scores)) if final_scores else 0.0,
    }
    write_json(output_dir / "metrics.json", payload)


def write_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
    skill_endpoint: ChatEndpoint,
) -> None:
    redacted_args = vars(args).copy()
    for key in ("policy_api_key", "skill_api_key"):
        if redacted_args.get(key):
            redacted_args[key] = "<redacted>"
    redacted_argv = list(sys.argv)
    for index, value in enumerate(redacted_argv):
        for flag in ("--policy-api-key", "--skill-api-key"):
            if value == flag and index + 1 < len(redacted_argv):
                redacted_argv[index + 1] = "<redacted>"
            elif value.startswith(f"{flag}="):
                redacted_argv[index] = f"{flag}=<redacted>"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "webshop_no_skill_validation",
        "project_root": str(PROJECT_ROOT),
        "argv": redacted_argv,
        "args": redacted_args,
        "policy_endpoint": {
            "base_url": policy_endpoint.base_url,
            "api_key": "<redacted>" if policy_endpoint.api_key else None,
            "model": policy_endpoint.model,
            "temperature": policy_endpoint.temperature,
            "max_completion_tokens": policy_endpoint.max_completion_tokens,
        },
        "skill_endpoint": {
            "base_url": skill_endpoint.base_url,
            "api_key": "<redacted>" if skill_endpoint.api_key else None,
            "model": skill_endpoint.model,
            "temperature": skill_endpoint.temperature,
            "max_completion_tokens": skill_endpoint.max_completion_tokens,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    prepare_output_dir(
        output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        regenerate_candidates=args.regenerate_candidates,
    )
    setup_logging(output_dir, args.log_level)

    policy_endpoint = resolve_endpoint(
        prefix="policy",
        args=args,
        default_base_url_env="POLICY_OPENAI_BASE_URL",
        default_model_env="POLICY_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.policy_temperature,
        max_completion_tokens=args.policy_max_completion_tokens,
        timeout=args.policy_timeout,
        retries=args.policy_retries,
        retry_delay=args.policy_retry_delay,
        extra_body_json=args.policy_extra_body_json,
    )
    skill_endpoint = resolve_endpoint(
        prefix="skill",
        args=args,
        default_base_url_env="SKILL_OPENAI_BASE_URL",
        default_model_env="SKILL_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.skill_temperature,
        max_completion_tokens=args.skill_max_completion_tokens,
        timeout=args.skill_timeout,
        retries=args.skill_retries,
        retry_delay=args.skill_retry_delay,
        extra_body_json=args.skill_extra_body_json,
    )
    write_run_config(
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
        skill_endpoint=skill_endpoint,
    )

    log_stage(output_dir, "task_sampling", "running")
    tasks = sample_tasks(args, output_dir)
    log_stage(
        output_dir,
        "task_sampling",
        "complete",
        sampled_tasks=len(tasks),
        rollouts_per_task=int(args.rollouts_per_task),
    )

    baseline_rollouts = collect_baseline_rollouts(
        tasks=tasks,
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
    )
    if args.stop_after_baseline_rollouts:
        log_stage(
            output_dir,
            "baseline_rollout",
            "complete",
            completed_rollouts=len(baseline_rollouts),
            stopped_before_skill_generation=True,
        )
        logging.info("Baseline rollouts complete; stopping before skill API generation.")
        return

    baseline_rollouts_for_generation = (
        baseline_rollouts[: max(0, int(args.max_candidates))]
        if args.max_candidates is not None
        else baseline_rollouts
    )

    candidate_skills = generate_candidate_skills(
        baseline_rollouts=baseline_rollouts_for_generation,
        output_dir=output_dir,
        skill_endpoint=skill_endpoint,
        max_candidates=None,
        max_workers=args.skill_gen_workers,
        resume=args.resume and not args.regenerate_candidates,
        include_episode_summary=args.include_episode_summary,
        skill_parse_attempts=args.skill_parse_attempts,
    )
    if args.stop_after_skill_generation:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(candidate_skills),
            parse_ok_skills=sum(1 for record in candidate_skills if record.get("parse_ok")),
            stopped_before_sft_export=True,
        )
        logging.info("Candidate skill API generation complete; stopping before SFT export.")
        return

    log_stage(output_dir, "sft_export", "running", candidate_skills=len(candidate_skills))
    sft_records = build_sft_exports_from_candidates(
        candidate_skills=candidate_skills,
        output_dir=output_dir,
        sft_val_ratio=args.sft_val_ratio,
        seed=args.seed,
        sft_min_source_score=args.sft_min_source_score,
        sft_include_success=args.sft_include_success,
        sft_max_zero_score_failures=args.sft_max_zero_score_failures,
        sft_max_records=args.sft_max_records,
        include_episode_summary=args.include_episode_summary,
    )
    log_stage(output_dir, "sft_export", "complete", sft_records=len(sft_records))

    write_metrics(
        tasks=tasks,
        baseline_rollouts=baseline_rollouts_for_generation,
        candidate_skills=candidate_skills,
        sft_records=sft_records,
        output_dir=output_dir,
    )
    log_stage(
        output_dir,
        "complete",
        "complete",
        sampled_tasks=len(tasks),
        baseline_rollouts=len(baseline_rollouts_for_generation),
        candidate_skills=len(candidate_skills),
        parse_ok_skills=sum(1 for record in candidate_skills if record.get("parse_ok")),
        sft_records=len(sft_records),
    )
    logging.info("WebShop episode-skill pipeline complete. Outputs are in %s.", output_dir)


if __name__ == "__main__":
    main()
