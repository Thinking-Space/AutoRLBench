#!/usr/bin/env python3
"""Collect visual EZPoints baseline rollouts through an OpenAI-compatible policy server."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from gym_cards.envs import EZPointEnv
from openai import OpenAI
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.prompts.gym_cards import GYM_CARDS_EZPOINTS_TEMPLATE


TASK_DESCRIPTION = (
    "Build a formula that evaluates to 12 using exactly the two card numbers shown, "
    "using each number once and appending one number, operator, or equals sign per step."
)
ACTION_VALUES = tuple([str(value) for value in range(1, 11)] + ["+", "*", "="])
ACTION_TO_ID = {value: index for index, value in enumerate(ACTION_VALUES)}
ACTION_PATTERN = re.compile(
    r'"action"\s*:\s*(?:"([^"]+)"|(-?\d+)|([^,}\]\s]+))',
    flags=re.IGNORECASE,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def response_text(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        text = str(content or "")
    reasoning = str(getattr(message, "reasoning_content", None) or "").strip()
    if reasoning and text and "<think>" not in text.lower():
        return f"<think>{reasoning}</think>\n{text}"
    return text or reasoning


def parse_action(text: str) -> Tuple[int, str, bool]:
    match = ACTION_PATTERN.search(str(text or ""))
    if not match:
        return -1, "", False
    action = next((value for value in match.groups() if value is not None), "").strip()
    if action not in ACTION_TO_ID:
        return -1, action, False
    return ACTION_TO_ID[action], action, True


def encode_and_save_image(image: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def build_prompt(info: Dict[str, Any]) -> str:
    formula = "".join(str(element) for element in info.get("Formula", []))
    return GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=formula)


class VisionPolicyAgent:
    def __init__(self, args: argparse.Namespace):
        self.model = args.policy_model
        self.temperature = float(args.policy_temperature)
        self.max_completion_tokens = int(args.policy_max_completion_tokens)
        self.retries = int(args.policy_retries)
        self.retry_delay = float(args.policy_retry_delay)
        self.client = OpenAI(
            api_key=args.policy_api_key,
            base_url=args.policy_base_url,
            timeout=float(args.policy_timeout),
            max_retries=0,
        )

    def get_action(self, prompt: str, image_url: str) -> Tuple[str, Optional[str]]:
        last_error: Optional[BaseException] = None
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt.replace("<image>", "the attached image")},
        ]
        for attempt in range(max(1, self.retries)):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    n=1,
                )
                return response_text(response).strip(), None
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_delay)
        fallback = json.dumps({"thoughts": "The policy request failed.", "action": "="})
        return fallback, f"{type(last_error).__name__}: {last_error}"


def collect_actions(
    *,
    agent: VisionPolicyAgent,
    prompts: Sequence[str],
    image_urls: Sequence[str],
    dones: Sequence[bool],
    workers: int,
) -> Tuple[List[str], List[Optional[str]]]:
    responses = [json.dumps({"thoughts": "The episode is done.", "action": "="}) for _ in prompts]
    errors: List[Optional[str]] = [None for _ in prompts]
    active_indices = [index for index, done in enumerate(dones) if not done]
    if not active_indices:
        return responses, errors

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(active_indices))) as executor:
        future_to_index = {
            executor.submit(agent.get_action, prompts[index], image_urls[index]): index
            for index in active_indices
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            responses[index], errors[index] = future.result()
    return responses, errors


def make_env(seed: int) -> Tuple[EZPointEnv, np.ndarray, Dict[str, Any]]:
    env = EZPointEnv()
    observation, info = env.reset(seed=seed)
    return env, observation, dict(info)


def run_wave(
    *,
    specs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    image_root: Path,
    agent: VisionPolicyAgent,
) -> List[Dict[str, Any]]:
    env_states = [make_env(int(spec["seed"])) for spec in specs]
    envs = [state[0] for state in env_states]
    images = [state[1] for state in env_states]
    infos = [state[2] for state in env_states]
    initial_infos = [json_safe(info) for info in infos]
    dones = [False] * len(specs)
    cumulative_rewards = np.zeros(len(specs), dtype=float)
    trajectories = [
        {
            "task_id": spec["task_id"],
            "task_type": "ezpoints",
            "goal_idx": int(spec["task_index"]),
            "sample_id": int(spec["task_index"]),
            "source_skill_id": None,
            "rollout_id": int(spec["rollout_id"]),
            "seed": int(spec["seed"]),
            "episode_skill": "",
            "task_description": TASK_DESCRIPTION,
            "initial_info": initial_infos[index],
            "steps": [],
        }
        for index, spec in enumerate(specs)
    ]

    for step_index in range(int(args.max_steps)):
        prompts = [build_prompt(info) for info in infos]
        image_urls: List[str] = []
        image_paths: List[Path] = []
        for index, spec in enumerate(specs):
            image_path = (
                image_root
                / f"task_{int(spec['task_index']):06d}"
                / f"rollout_{int(spec['rollout_id']):03d}"
                / f"step_{step_index:03d}.png"
            )
            image_paths.append(image_path)
            image_urls.append(encode_and_save_image(images[index], image_path) if not dones[index] else "")

        responses, errors = collect_actions(
            agent=agent,
            prompts=prompts,
            image_urls=image_urls,
            dones=dones,
            workers=int(args.request_workers),
        )
        action_data = [parse_action(response) for response in responses]

        for index, env in enumerate(envs):
            if dones[index]:
                continue
            action_id, action_value, action_valid = action_data[index]
            next_image, reward, terminated, truncated, info = env.step(action_id)
            done = bool(terminated or truncated)
            cumulative_rewards[index] += float(reward)
            info = dict(info)
            info["is_action_valid"] = bool(action_valid)
            info["score"] = float(cumulative_rewards[index])
            if errors[index]:
                info["policy_error"] = errors[index]
            next_prompt = "" if done else build_prompt(info)
            trajectories[index]["steps"].append(
                {
                    "step_idx": step_index,
                    "observation": prompts[index],
                    "observation_prompt": prompts[index],
                    "skill_augmented_observation": prompts[index],
                    "model_response": responses[index],
                    "raw_action_text": responses[index],
                    "executed_action": action_value,
                    "action_id": int(action_id),
                    "action_valid": bool(action_valid),
                    "reward": float(reward),
                    "score": float(cumulative_rewards[index]),
                    "done": done,
                    "info": json_safe(info),
                    "images": [{"image": str(image_paths[index].resolve())}],
                    "next_observation": next_prompt,
                    "next_observation_prompt": next_prompt,
                }
            )
            images[index] = next_image
            infos[index] = info
            dones[index] = done

        if all(dones):
            break

    for index, trajectory in enumerate(trajectories):
        success = bool(infos[index].get("won", False) or cumulative_rewards[index] > 0)
        trajectory["success"] = success
        trajectory["completed"] = bool(dones[index])
        trajectory["num_steps"] = len(trajectory["steps"])
        trajectory["final_reward"] = float(cumulative_rewards[index])
        trajectory["final_task_score"] = float(cumulative_rewards[index])
        trajectory["final_info"] = json_safe(infos[index])
        trajectory["baseline_key"] = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
    return trajectories


def write_summary(path: Path, records: Sequence[Dict[str, Any]], expected: int) -> None:
    keys = [str(record["baseline_key"]) for record in records]
    missing_images = sum(
        int(not Path(str(image["image"])).exists())
        for record in records
        for step in record.get("steps", [])
        for image in step.get("images", [])
    )
    successes = sum(int(bool(record.get("success", False))) for record in records)
    payload = {
        "records": len(records),
        "expected_records": expected,
        "tasks": len({str(record["task_id"]) for record in records}),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "step_records": sum(len(record.get("steps", [])) for record in records),
        "unique_baseline_keys": len(set(keys)),
        "missing_images": missing_images,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--num-tasks", type=int, default=180)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--task-batch-size", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--request-workers", type=int, default=64)
    parser.add_argument("--policy-base-url", default="http://127.0.0.1:60004/v1")
    parser.add_argument("--policy-api-key", default="EMPTY")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=512)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--policy-retries", type=int, default=2)
    parser.add_argument("--policy-retry-delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else output_dir / "baseline_rollouts.jsonl"
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = baseline_path.with_suffix(".summary.json")
    sampled_tasks_path = output_dir / "sampled_tasks.jsonl"
    image_root = output_dir / "ezpoints_images"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "baseline_rollouts.log", encoding="utf-8"),
        ],
    )
    for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if args.overwrite:
        for path in (baseline_path, summary_path, sampled_tasks_path):
            path.unlink(missing_ok=True)
        shutil.rmtree(image_root, ignore_errors=True)

    tasks = [
        {
            "task_id": f"ezpoints_{task_index:06d}",
            "task_type": "ezpoints",
            "goal_idx": task_index,
            "sample_id": task_index,
            "seed": int(args.seed) + task_index,
            "task_description": TASK_DESCRIPTION,
        }
        for task_index in range(int(args.num_tasks))
    ]
    if not sampled_tasks_path.exists():
        for task in tasks:
            append_jsonl(sampled_tasks_path, task)

    existing = read_jsonl(baseline_path) if args.resume else []
    existing_keys = {(str(record["task_id"]), int(record["rollout_id"])) for record in existing}
    pending = [
        {
            "task_id": task["task_id"],
            "task_index": int(task["goal_idx"]),
            "rollout_id": rollout_id,
            "seed": int(task["seed"]),
        }
        for task in tasks
        for rollout_id in range(int(args.rollouts_per_task))
        if (str(task["task_id"]), rollout_id) not in existing_keys
    ]
    expected = int(args.num_tasks) * int(args.rollouts_per_task)
    wave_size = int(args.task_batch_size) * int(args.rollouts_per_task)
    total_waves = (len(pending) + wave_size - 1) // wave_size
    logging.info(
        "Collecting %d pending visual EZPoints rollouts (%d existing, %d expected) in %d wave(s).",
        len(pending),
        len(existing),
        expected,
        total_waves,
    )

    agent = VisionPolicyAgent(args)
    records = list(existing)
    for wave_index, specs in enumerate(chunked(pending, wave_size), start=1):
        logging.info(
            "Baseline wave %d/%d: %d environments across %d tasks.",
            wave_index,
            total_waves,
            len(specs),
            len({int(spec["task_index"]) for spec in specs}),
        )
        wave_records = run_wave(specs=specs, args=args, image_root=image_root, agent=agent)
        for record in wave_records:
            append_jsonl(baseline_path, record)
            records.append(record)
        write_summary(summary_path, records, expected)
        logging.info("Completed %d/%d baseline rollouts.", len(records), expected)

    keys = [str(record["baseline_key"]) for record in records]
    if len(records) != expected:
        raise SystemExit(f"Expected {expected} baseline rollouts, found {len(records)}")
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate baseline_key values found")
    write_summary(summary_path, records, expected)
    logging.info("Visual EZPoints baseline collection complete: %s", summary_path)


if __name__ == "__main__":
    main()
