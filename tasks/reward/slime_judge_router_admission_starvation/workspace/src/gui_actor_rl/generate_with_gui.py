"""GUI actor RL rollout with a frozen reward-agent (judge) as the reward.

Adapted from OpenClaw-RL/gui-rl ``generate_with_gui.py``:
- env access goes through guirl's env_service (env master :9900, docker VMs)
  via :class:`gui_actor_rl.env_client.ActorEnvClient`;
- the PRM (process reward model) path is removed entirely;
- the training reward comes from the frozen judge (reward agent) instead of
  the environment's rule-based ``evaluate()``:
    * judge says "success"  -> reward +1.0
    * judge says "failure"  -> reward 0.0
    * "uncertain" / "parse_failure" -> sample is DROPPED from training
      (``remove_sample=True``);
- the rule-based env score is still computed and logged to
  ``metadata["gui_score"]`` / reward key ``acc`` purely for monitoring
  (judge-vs-env agreement is exposed as reward key ``judge_env_agree``).

All gui_*/judge_* settings are injected onto ``args`` via slime's
``--custom-config-path`` YAML (the fork's argparse does not know these flags).
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import faulthandler
import json
import logging
import os
import re
import shutil
import sys
import random
import time
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.misc import load_function
from slime.utils.types import Sample

from gui_actor_rl import mm_preprocess
from gui_actor_rl.agents.qwen3vl_agent import process_image
from gui_actor_rl.env_client import ActorEnvClient
from gui_actor_rl.action_execution import action_chunks
from gui_actor_rl.judge_client import JudgeClient, JudgeResult
from gui_actor_rl.probe_client import RolloutMasterProbeRunner

# Judge->probe->judge state machine, reused from the reward-agent (guirl) code.
# ``run_judge_inference`` replicates guirl.slime_custom.custom_generate's judge
# loop (inference-only) so the FROZEN judge sees the exact input distribution it
# was trained on. The probe runs through the SAME host-side rollout master +
# evaluation_harness holo3 harness used during judge training.
from gui_actor_rl.judge_infer import run_judge_inference

# Passive ZeroGUI judge (gui_reward_source == "zerogui"). Mutually exclusive
# with the reward-agent judge stack above. Imported lazily inside
# ``_run_zerogui_judge`` because gui_actor_rl.zerogui_judge pulls in the
# vendored ``mm_agents`` package (external/pkgs), which the judge/env reward
# paths never need.


async def _run_zerogui_judge(**kwargs):
    from gui_actor_rl.zerogui_judge import run_zerogui_judge

    return await run_zerogui_judge(**kwargs)

# Gated wall-clock instrumentation (GUI_TIMING_LOG_ENABLE / gui_timing_log_enable).
# When disabled, every recorder call is a no-op.
from gui_actor_rl.timing import (
    SampleTimingRecorder,
    set_current_rollout_id,
    write_rollout_summary,
)

logger = logging.getLogger(__name__)
faulthandler.enable(file=sys.stderr, all_threads=True)

_ANSI_RESET = "\033[0m"
_ANSI_COLORS = {
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "none": "",
}

# GUI_RL_PROJECT project root: .../GUI_RL_PROJECT/third_party/slime/gui_actor_rl
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_BASE_DIR = str(_PROJECT_ROOT / "evaluation_examples")
_DEFAULT_TRAIN_META = "test_meta_noproxy_nogdrive_feasible_offline280_train_nomulti.json"
# Offline-only eval split; training scripts do not set GUI_EVAL_META_PATH / eval-interval.
_DEFAULT_EVAL_META = "test_meta_noproxy_nogdrive_feasible_offline280_eval_multiapp.json"


def _gui_log(message: str, *args: Any) -> None:
    color_name = os.getenv("GUI_LOG_COLOR", "yellow").strip().lower()
    color_code = _ANSI_COLORS.get(color_name, _ANSI_COLORS["yellow"])
    prefix = "[GUI-ACTOR]"
    if color_code:
        prefix = f"{color_code}{prefix}{_ANSI_RESET}"
    logger.info(f"{prefix} {message}", *args)


@lru_cache(maxsize=1)
def _get_gui_trajectory_semaphore() -> asyncio.Semaphore:
    max_envs = max(1, int(os.getenv("GUI_POOL_MAX_ENVS", "16")))
    concurrency = max(1, int(os.getenv("GUI_TRAJECTORY_CONCURRENCY", str(max_envs))))
    return asyncio.Semaphore(concurrency)


@lru_cache(maxsize=1)
def _get_task_clear_lock() -> asyncio.Lock:
    return asyncio.Lock()


_RESULT_RUN_ID = uuid.uuid4().hex


_JUDGE_CLIENT: JudgeClient | None = None


def _gui_reward_source(args: Any) -> str:
    """Reward source switch (mutually exclusive, n-choose-one).

    Arg / env name is kept identical to upstream (``gui_reward_source`` /
    ``GUI_REWARD_SOURCE``) so this fork stays merge-compatible with origin. We
    extend upstream's ``{judge, env}`` with a third value, ``zerogui``:

    - ``"judge"`` (default): frozen reward agent (judge) + holo3 probe loop
      (``_run_judge_with_probe`` -> gui_actor_rl.judge_infer). This is what the
      project calls the "reward agent". Needs the judge sglang service AND the
      host-side rollout master for the probe.
    - ``"env"``: OSWorld's rule-based ``evaluate()`` score, straight through —
      no judge server, no probe, no rollout master (ported from upstream).
    - ``"zerogui"``: passive ZeroGUI judge (gui_actor_rl.zerogui_judge). Scores
      the finished trajectory from screenshots in one pass; no probe, no live
      lease, no judge service — only a ZeroGUI VLM endpoint.

    Exactly one runs per rollout. Everything downstream (``_judge_reward`` /
    reward / metadata / ``reward_post_process``) is shared: ``judge`` and
    ``zerogui`` both produce a ``JudgeResult``; ``env`` writes the reward dict
    directly.
    """
    src = (
        getattr(args, "gui_reward_source", None)
        or os.getenv("GUI_REWARD_SOURCE", "judge")
    )
    src = str(src).strip().lower()
    if src not in {"judge", "env", "zerogui"}:
        raise ValueError(f"gui_reward_source must be 'judge', 'env' or 'zerogui', got: {src}")
    return src


def _get_judge_client(args: Any) -> JudgeClient:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = JudgeClient.from_args(args)
        _gui_log(
            "Judge client initialized: url=%s model=%s",
            _JUDGE_CLIENT.base_url,
            _JUDGE_CLIENT.model_name,
        )
    return _JUDGE_CLIENT


def _probe_enabled(args: Any) -> bool:
    return bool(getattr(args, "gui_probe_enable", True))


async def _close_lease_bounded(env_client: Any, lease_id: str, timing: Any) -> None:
    """Release an env lease with a wall-clock cap (GUI_ENV_CLOSE_TIMEOUT_S)."""
    close_timeout_s = float(os.getenv("GUI_ENV_CLOSE_TIMEOUT_S", "120"))
    with timing.phase("env_close"):
        await asyncio.wait_for(env_client.close(lease_id=lease_id), timeout=close_timeout_s)


# Logical markers from the actor (WAIT/DONE/FAIL). Same contract as
# evaluation_harness.markers / guirl actor_client: NEVER forward to env.step —
# DONE/FAIL would set OSWorldEnv._terminated and break the later judge probe
# on the same lease; WAIT is a no-op that only needs sleep + get_obs.
_LOGICAL_MARKERS = frozenset({"WAIT", "DONE", "FAIL"})


def _logical_marker(action: Any) -> str | None:
    if not isinstance(action, str):
        return None
    marker = action.strip().upper()
    return marker if marker in _LOGICAL_MARKERS else None


async def _handle_logical_marker(
    env_client: ActorEnvClient,
    *,
    lease_id: str,
    marker: str,
    sleep_after_execution: float,
    current_obs: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Fake-execute WAIT/DONE/FAIL without touching env.step.

    Returns ``(obs, done)``. WAIT keeps the episode open; DONE/FAIL end it.
    """
    if sleep_after_execution and sleep_after_execution > 0:
        await asyncio.sleep(float(sleep_after_execution))
    obs = current_obs
    try:
        obs = await env_client.get_obs(lease_id)
    except Exception:
        logger.warning("get_obs after logical marker %s failed for lease %s", marker, lease_id, exc_info=True)
    return obs, marker in {"DONE", "FAIL"}


def _build_probe_runner(args: Any, result_dir: Path) -> RolloutMasterProbeRunner | None:
    """Construct the rollout-master probe runner (same harness used to TRAIN
    the judge). The probe navigates the live env lease to gather per-criterion
    evidence via the host-side rollout master + evaluation_harness holo3 harness.
    holo3 is a fixed external endpoint; the rollout master runs host-side.
    """
    return RolloutMasterProbeRunner.from_args(
        args, example_result_dir=str(result_dir / "probe")
    )


async def _run_judge_with_probe(
    *,
    args: Any,
    judge_client: JudgeClient,
    result_dir: Path,
    lease_id: str | None,
    fence_token: str | None,
    instruction: str,
    screenshots_b64: list[str],
    action_history: list[str] | None,
    timing_records: list[dict[str, Any]] | None = None,
) -> JudgeResult:
    """Judge->probe->judge loop (frozen judge), aligned to guirl training.

    Delegates to ``gui_actor_rl.judge_infer.run_judge_inference``, which
    replicates ``guirl.slime_custom.custom_generate``'s judge state machine
    (multi-round loop, strip-thinking history, uncertain->probe->re-judge, parse
    retry). Must be called while the env lease is still open — the holo3 probe
    steps the SAME env to gather evidence before re-judging. Never raises;
    parse/probe failures degrade to a verdict or ``status="parse_failure"``.
    """
    if not screenshots_b64:
        return JudgeResult(status="parse_failure", error="empty trajectory (no screenshots)")

    history_n = int(getattr(args, "gui_judge_history_n", 3) or 3)
    max_rounds = int(getattr(args, "gui_judge_max_rounds", 3) or 3)

    probe_runner: RolloutMasterProbeRunner | None = None
    if _probe_enabled(args) and lease_id is not None:
        probe_runner = _build_probe_runner(args, result_dir)

    try:
        return await run_judge_inference(
            judge_client=judge_client,
            probe_runner=probe_runner,
            lease_id=lease_id,
            fence_token=fence_token,
            instruction=instruction,
            trajectory_screenshots=list(screenshots_b64),
            action_history=list(action_history) if action_history else None,
            max_rounds=max_rounds,
            history_n=history_n,
            timing_records=timing_records,
        )
    except Exception as exc:
        logger.warning("judge+probe loop failed: %s", exc, exc_info=True)
        return JudgeResult(status="parse_failure", error=str(exc))
    finally:
        if probe_runner is not None:
            try:
                await probe_runner.close_session()
            except Exception:
                logger.debug("probe client session close failed", exc_info=True)


@lru_cache(maxsize=16)
def _load_meta_pairs(meta_path: str) -> list[tuple[str, str]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    pairs: list[tuple[str, str]] = []
    for domain, examples in meta.items():
        for example_id in examples:
            pairs.append((str(domain), str(example_id)))

    shuffle_seed_str = os.getenv("GUI_TASK_SHUFFLE_SEED", "42")
    if shuffle_seed_str.strip().lower() not in {"", "none", "-1"}:
        import random

        rng = random.Random(int(shuffle_seed_str))
        rng.shuffle(pairs)
        _gui_log("Shuffled %d task pairs with seed=%s", len(pairs), shuffle_seed_str)

    return pairs


def _load_task_config(base_dir: str, domain: str, example_id: str) -> dict[str, Any]:
    cfg_path = Path(base_dir) / "examples" / domain / f"{example_id}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_media_artifacts(args: Any) -> bool:
    """gui_trajectory_artifacts: "full" (default) keeps step_N.png and trajectory.json (the whole
    chat history with base64 images re-embedded at every step, ~80 MB per episode); "text" writes
    only traj.jsonl, result.txt and judge.json. Nothing in training reads the dropped files."""
    mode = str(getattr(args, "gui_trajectory_artifacts", "full") or "full").lower()
    if mode not in ("full", "text"):
        raise ValueError(f"gui_trajectory_artifacts must be full or text, got {mode!r}")
    return mode == "full"


def _save_image(image_bytes: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)


def _write_json(path: Path, data: Any, *, append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=None if append else 2)
        if append:
            f.write("\n")


def _sample_task_info(sample: Sample, evaluation: bool = False) -> tuple[str, dict[str, Any] | None, str, str]:
    metadata = sample.metadata or {}
    instruction = metadata.get("instruction")
    task_config = metadata.get("task_config")
    domain = metadata.get("domain", "default")
    example_id = metadata.get(
        "example_id", f"sample_{sample.index if sample.index is not None else uuid.uuid4().hex[:8]}"
    )

    if not instruction and isinstance(sample.prompt, str):
        prompt = sample.prompt.strip()
        if prompt.startswith("{") and prompt.endswith("}"):
            try:
                obj = json.loads(prompt)
                instruction = obj.get("instruction") or instruction
                task_config = obj.get("task_config") or task_config
                domain = obj.get("domain", domain)
                example_id = obj.get("example_id", example_id)
            except Exception:
                instruction = prompt
        else:
            instruction = prompt

    if task_config is None:
        base_dir = os.getenv("GUI_TEST_CONFIG_BASE_DIR", _DEFAULT_CONFIG_BASE_DIR)
        default_meta = _DEFAULT_EVAL_META if evaluation else _DEFAULT_TRAIN_META
        env_meta_key = "GUI_EVAL_META_PATH" if evaluation else "GUI_TRAIN_META_PATH"
        meta_path = os.getenv(env_meta_key, str(Path(base_dir) / default_meta))
        if isinstance(sample.prompt, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", sample.prompt):
            example_id = sample.prompt
        else:
            try:
                pairs = _load_meta_pairs(meta_path)
                sample_key = sample.group_index if sample.group_index is not None else sample.index or 0
                if pairs:
                    domain, example_id = pairs[int(sample_key) % len(pairs)]
            except Exception:
                logger.exception("Failed to resolve task pair from meta file: %s", meta_path)

        try:
            task_config = _load_task_config(base_dir, domain, example_id)
            if not instruction:
                instruction = str(task_config.get("instruction", ""))
        except Exception:
            logger.exception("Failed to load task config for %s/%s", domain, example_id)

    return instruction or "", task_config, str(domain), str(example_id)


def _build_result_dir(args: Any, domain: str, example_id: str, sample: Sample) -> Path:
    base = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    action_space = os.getenv("GUI_ACTION_SPACE", "pyautogui")
    observation_type = os.getenv("GUI_OBSERVATION_TYPE", "screenshot")
    model_name = getattr(args, "hf_checkpoint", "gui-policy-model")
    model_tag = re.sub(r"[\\/]+", "_", str(model_name)).lstrip("_")
    # IMPORTANT: each sample must have an isolated output dir.
    if sample.index is not None:
        run_idx = f"s{sample.index}"
    elif sample.group_index is not None:
        run_idx = f"g{sample.group_index}"
    else:
        run_idx = f"u{uuid.uuid4().hex[:8]}"
    out = base / action_space / observation_type / model_tag / domain / example_id / run_idx
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_task_dir(args: Any, domain: str, example_id: str) -> Path:
    base = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    action_space = os.getenv("GUI_ACTION_SPACE", "pyautogui")
    observation_type = os.getenv("GUI_OBSERVATION_TYPE", "screenshot")
    model_name = getattr(args, "hf_checkpoint", "gui-policy-model")
    model_tag = re.sub(r"[\\/]+", "_", str(model_name)).lstrip("_")
    task_dir = base / action_space / observation_type / model_tag / domain / example_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


async def _maybe_clear_task_result_dir(args: Any, domain: str, example_id: str, sample: Sample) -> None:
    if os.getenv("GUI_CLEAR_TASK_RESULT_ON_START", "1").strip().lower() in {"0", "false", "no"}:
        return

    def clear_once():
        task_dir = _build_task_dir(args, domain, example_id)
        marker = task_dir / ".gui_actor_run"
        if marker.exists() and marker.read_text() == _RESULT_RUN_ID:
            return
        shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(_RESULT_RUN_ID)
        _gui_log("cleared task result dir for %s", task_dir)

    # One disk marker per task replaces the unbounded per-group history set.
    # Never delete another group's active output for the same task.
    async with _get_task_clear_lock():
        task = asyncio.create_task(asyncio.to_thread(clear_once))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep the lock until the filesystem mutation actually finishes.
            await task
            raise


def _create_gui_agent(args: Any, *, max_steps: int, max_image_history_length: int, result_dir: Path):
    agent_cls_path = (
        getattr(args, "gui_agent_class_path", None)
        or os.getenv("GUI_AGENT_CLASS_PATH", "gui_actor_rl.agents.qwen3vl_agent.Qwen3VLAgentLocal")
    )
    agent_cls = load_function(agent_cls_path)
    return agent_cls(
        model=getattr(args, "hf_checkpoint", "gui-policy-model"),
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        action_space=os.getenv("GUI_ACTION_SPACE", "pyautogui"),
        observation_type=os.getenv("GUI_OBSERVATION_TYPE", "screenshot"),
        coordinate_type=os.getenv("GUI_COORDINATE_TYPE", "relative"),
        example_result_dir=str(result_dir),
    )


def _judge_reward(judge_result: JudgeResult, eval_score: float) -> tuple[dict[str, float], bool]:
    """Map judge verdict to (reward dict, drop_sample).

    success -> 1.0; failure -> 0.0; uncertain/parse_failure -> dropped.
    (0/1 reward: GRPO group normalization subtracts the group mean, so only
    the relative gap matters; dropped samples also carry score=0.0 but are
    distinguished by remove_sample=True and excluded from the baseline in
    reward_post_process.)
    The probe (holo3) resolves "uncertain" in-loop where it can, so a dropped
    sample here means uncertain *even after* probing (or a parse failure).
    ``acc`` carries the rule-based env score (monitoring only);
    ``judge_env_agree`` is 1.0 when the judge verdict matches env evaluate;
    ``judge_used_probe`` flags whether the probe fired (wandb monitoring).
    """
    status = judge_result.status
    # eval_score < 0 means env evaluate() itself failed (judge reward is still
    # valid): report acc as 0 and no agreement rather than leaking -1 into the
    # wandb means.
    env_eval_failed = float(eval_score) < 0.0
    acc = max(float(eval_score), 0.0)
    env_success = float(eval_score) == 1.0
    used_probe = 1.0 if judge_result.used_probe else 0.0
    if status == "success":
        agree = 0.0 if env_eval_failed else (1.0 if env_success else 0.0)
        return {"score": 1.0, "acc": acc, "judge_env_agree": agree, "judge_used_probe": used_probe}, False
    if status == "failure":
        agree = 0.0 if env_eval_failed else (0.0 if env_success else 1.0)
        return {"score": 0.0, "acc": acc, "judge_env_agree": agree, "judge_used_probe": used_probe}, False
    # uncertain / parse_failure: no usable training signal, drop the sample.
    return {"score": 0.0, "acc": acc, "judge_env_agree": 0.0, "judge_used_probe": used_probe}, True


def _rollout_log_probs_of(train_data: dict[str, Any] | None) -> list[float]:
    """Engine log-probs aligned 1:1 with the response tokens of one generation step.

    slime builds the rollout batch from ``samples[0].rollout_log_probs is not None`` and then
    reads that field from EVERY sample, so each sample must carry a list (``[]`` for a sample
    without response tokens). The length check here (and the one in slice_log_prob_with_cp)
    keeps the mismatch metrics from silently pairing log-probs with the wrong tokens.
    """
    if train_data is None:
        return []
    log_probs = train_data["response_log_probs"]
    response_length = int(train_data["response_length"])
    if len(log_probs) != response_length:
        raise RuntimeError(
            f"rollout log-prob count {len(log_probs)} != response_length {response_length}; "
            "refusing to build a sample whose log-probs are misaligned with its tokens"
        )
    return list(log_probs)


def _prefix_cache_info_of(train_data: dict[str, Any] | None) -> Sample.PrefixCacheInfo:
    timing = (train_data or {}).get("gen_timing") or {}
    return Sample.PrefixCacheInfo(
        cached_tokens=int(timing.get("cached_tokens") or 0),
        total_prompt_tokens=int(timing.get("prompt_tokens") or 0),
    )


def _build_dynamic_history_samples(
    args: Any,
    state: GenerateState,
    agent: Any,
    base_sample: Sample,
    step_snapshots: list[dict[str, Any]],
    outcome_reward: dict[str, float] | None,
) -> list[Sample]:
    dynamic_samples: list[Sample] = []

    for snapshot in step_snapshots:
        step_idx = int(snapshot["step_idx"])
        response_text = snapshot["response_text"]
        # Token passthrough (slime-core semantics): tokens/loss_mask/mm were
        # captured AT GENERATION TIME by generate_with_sglang — the training
        # sequence IS the generation sequence, never re-rendered. loss_mask is
        # [1]*response_length by construction (history is prompt, so it is
        # masked out by position, not by re-render bookkeeping).
        train_data = snapshot["train_data"]
        token_ids = train_data["tokens"]
        response_length = int(train_data["response_length"])
        if response_length <= 0:
            continue
        child_loss_mask = list(train_data["loss_mask"])
        mm_train = train_data["mm_train"]
        rollout_log_probs = _rollout_log_probs_of(train_data)

        child_metadata = copy.deepcopy(base_sample.metadata or {})
        child_metadata["dynamic_step_index"] = step_idx

        child = Sample(
            group_index=base_sample.group_index,
            # All per-step children of one trajectory form one training
            # aggregation group so loss averaging uses the whole-trajectory
            # denominator (see Sample.group_id in slime/utils/types.py).
            group_id=base_sample.index,
            index=base_sample.index,
            prompt=base_sample.prompt,
            tokens=token_ids,
            multimodal_inputs=base_sample.multimodal_inputs,
            multimodal_train_inputs=mm_train,
            response=response_text,
            response_length=response_length,
            label=base_sample.label,
            reward=copy.deepcopy(outcome_reward),
            loss_mask=child_loss_mask,
            weight_versions=list(base_sample.weight_versions),
            rollout_log_probs=rollout_log_probs,
            rollout_routed_experts=None,
            remove_sample=base_sample.remove_sample,
            status=base_sample.status,
            metadata=child_metadata,
            generate_function_path=base_sample.generate_function_path,
            train_metadata=base_sample.train_metadata,
            non_generation_time=base_sample.non_generation_time,
            spec_info=base_sample.spec_info,
            prefix_cache_info=_prefix_cache_info_of(train_data),
        )
        dynamic_samples.append(child)

    return dynamic_samples


_RETRY_EPISODE = object()


async def generate(
    args,
    sample: Sample,
    sampling_params,
    evaluation: bool = False,
    _episode_attempt: int = 0,
) -> Sample | list[Sample]:
    # Finish the old coroutine before retrying. Recursive await retained its
    # screenshots, step snapshots and plasma views throughout the new episode.
    retry_history = []
    while True:
        result = await _generate_attempt(args, sample, sampling_params, evaluation, _episode_attempt, retry_history)
        if result is not _RETRY_EPISODE:
            return result
        _episode_attempt += 1


async def _generate_attempt(
    args,
    sample: Sample,
    sampling_params,
    evaluation: bool = False,
    _episode_attempt: int = 0,
    _retry_history: list | None = None,
) -> Sample | list[Sample]:
    assert not args.partial_rollout, "Partial rollout is not supported for GUI rollout."
    attempt_start = time.monotonic()

    instruction, task_config, domain, example_id = await asyncio.to_thread(
        _sample_task_info, sample, evaluation=evaluation
    )
    await _maybe_clear_task_result_dir(args, domain, example_id, sample)
    result_dir = await asyncio.to_thread(_build_result_dir, args, domain, example_id, sample)
    traj_path = result_dir / "traj.jsonl"

    # Gated timing recorder (no-op unless gui_timing_log_enable / GUI_TIMING_LOG_ENABLE).
    timing = SampleTimingRecorder(
        args,
        sample_index=sample.index,
        group_index=sample.group_index,
        domain=domain,
        example_id=example_id,
        instruction=instruction,
        evaluation=evaluation,
    )

    env_url = (
        getattr(args, "gui_env_master_url", None)
        or os.getenv("GUI_ENV_MASTER_URL", "http://host_0:9900")
    )
    env_client = ActorEnvClient(env_url)
    # Reward source switch (n-choose-one: judge / env / zerogui). The reward-agent
    # judge service is only initialized on the "judge" path; env and zerogui
    # never touch it.
    reward_source = _gui_reward_source(args)
    judge_client = _get_judge_client(args) if reward_source == "judge" else None
    state = GenerateState(args)
    # Multi-core preprocessing pool (gui_mm_pool_size > 0): offloads the HF
    # processor CPU work to Ray actors. Idempotent; no-op when disabled.
    await mm_preprocess.ensure_mm_pool(args)

    rollout_max_steps = getattr(args, "gui_max_steps", None)
    if rollout_max_steps is None:
        rollout_max_steps = int(os.getenv("GUI_MAX_STEPS", "15"))

    eval_max_steps = getattr(args, "gui_eval_max_steps", None)
    max_steps = int(eval_max_steps if evaluation and eval_max_steps is not None else rollout_max_steps)

    rollout_sleep = getattr(args, "gui_sleep_after_execution", None)
    if rollout_sleep is None:
        rollout_sleep = float(os.getenv("GUI_SLEEP_AFTER_EXECUTION", "0.0"))

    eval_sleep = getattr(args, "gui_eval_sleep_after_execution", None)
    sleep_after_execution = float(eval_sleep if evaluation and eval_sleep is not None else rollout_sleep)

    rollout_wait_reset = getattr(args, "gui_wait_after_reset", None)
    if rollout_wait_reset is None:
        rollout_wait_reset = float(os.getenv("GUI_WAIT_AFTER_RESET", "0.0"))
    eval_wait_reset = getattr(args, "gui_eval_wait_after_reset", None)
    wait_after_reset = float(eval_wait_reset if evaluation and eval_wait_reset is not None else rollout_wait_reset)

    # Hard wall-clock budget per trajectory (env allocate -> ... -> judge/probe).
    # Long-tail trajectories otherwise stall the whole rollout: generate_rollout
    # waits for EVERY submitted group, so one stuck episode delays training by
    # its full duration. On expiry the sample is ABORTED (-> remove_sample=True,
    # excluded from training AND from the GRPO baseline in reward_post_process)
    # and the env lease is released in ``finally``. <=0 disables the timeout.
    # Eval defaults to disabled so timeouts never skew eval accuracy.
    rollout_traj_timeout = getattr(args, "gui_trajectory_timeout_s", None)
    if rollout_traj_timeout is None:
        rollout_traj_timeout = float(os.getenv("GUI_TRAJECTORY_TIMEOUT_S", "2400"))
    eval_traj_timeout = getattr(args, "gui_eval_trajectory_timeout_s", None)
    if eval_traj_timeout is None:
        eval_traj_timeout = float(os.getenv("GUI_EVAL_TRAJECTORY_TIMEOUT_S", "0"))
    trajectory_timeout_s = float(eval_traj_timeout if evaluation else rollout_traj_timeout)

    max_image_history_cfg = getattr(args, "gui_max_image_history_length", None)
    if max_image_history_cfg is None:
        max_image_history_cfg = int(os.getenv("GUI_MAX_IMAGE_HISTORY_LENGTH", str(max_steps)))
    max_image_history_length = int(max_image_history_cfg)

    sampling_params = dict(sampling_params)
    if evaluation and getattr(args, "eval_temperature", None) is not None:
        sampling_params["temperature"] = float(args.eval_temperature)
    elif (not evaluation) and getattr(args, "rollout_temperature", None) is not None:
        sampling_params["temperature"] = float(args.rollout_temperature)

    parser = _create_gui_agent(
        args,
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        result_dir=result_dir,
    )
    parser.reset(logging.getLogger("desktopenv.gui_agent.rollout"))

    assistant_responses: list[str] = []
    step_snapshots: list[dict[str, Any]] = []
    last_step_train_data_for_loss: dict[str, Any] | None = None
    trace_records: list[dict[str, Any]] = []
    save_media = _save_media_artifacts(args)
    final_status = Sample.Status.COMPLETED
    eval_score = 0.0
    final_screenshot_b64: str | None = None
    judge_result: JudgeResult | None = None
    # Only the text is needed after logging. Retaining an exception also keeps
    # its traceback and this coroutine's image buffers in a reference cycle.
    rollout_error: str | None = None
    retryable_env_failure = False
    rollout_phase = "setup"

    trajectory_semaphore = _get_gui_trajectory_semaphore()
    with timing.phase("semaphore_wait"):
        await trajectory_semaphore.acquire()
    semaphore_released = False
    lease_id: str | None = None
    # Deadline covers env allocate -> steps -> evaluate -> judge/probe, but NOT
    # the semaphore wait above (queueing is not the trajectory's own time).
    # asyncio.timeout cancels this coroutine at expiry; the lease release in
    # ``finally`` below still runs, so the VM goes back to the pool immediately.
    trajectory_deadline = (
        asyncio.timeout(trajectory_timeout_s)
        if trajectory_timeout_s > 0
        else contextlib.nullcontext()
    )
    timed_out = False
    try:
        async with trajectory_deadline:
            _gui_log(
                "GUI rollout start sample=%s group=%s domain=%s example=%s eval=%s",
                sample.index,
                sample.group_index,
                domain,
                example_id,
                evaluation,
            )
            # GUI env capacity is usually the bottleneck. Retry allocation so
            # temporary pool exhaustion does not hard-crash rollout workers.
            rollout_phase = "env_allocate"
            allocate_retries = int(os.getenv("GUI_ALLOCATE_RETRIES", "10"))
            allocate_backoff_seconds = float(os.getenv("GUI_ALLOCATE_BACKOFF_SECONDS", "2.0"))
            allocate_error: Exception | None = None
            with timing.phase("env_allocate"):
                episode_id = f"{domain}:{example_id}:{uuid.uuid4().hex}"
                for attempt in range(allocate_retries):
                    try:
                        lease = await env_client.allocate(episode_id=episode_id)
                        lease_id = lease["lease_id"]
                        allocate_error = None
                        break
                    except Exception as e:  # pragma: no cover - external service call
                        allocate_error = e
                        if attempt < allocate_retries - 1:
                            logger.warning(
                                "GUI allocate failed (%d/%d), will retry in %.1fs: %s",
                                attempt + 1,
                                allocate_retries,
                                allocate_backoff_seconds,
                                e,
                            )
                            await asyncio.sleep(allocate_backoff_seconds)
            if lease_id is None:
                raise RuntimeError(
                    f"GUI env allocate failed after {allocate_retries} retries from {env_url}: {allocate_error}"
                )

            rollout_phase = "env_reset"
            with timing.phase("env_reset"):
                obs = await env_client.reset(lease_id=lease_id, task_config=task_config)
                if wait_after_reset > 0:
                    # Spread the synchronized wake-up across a rollout: every
                    # trajectory sleeps wait_after_reset + uniform(0, JITTER_S).
                    # A one-time jitter at reset is enough — the natural variance
                    # in step duration (gen_s σ≈6s) keeps trajectories from
                    # re-synchronising after the initial phase is scattered.
                    # Adding per-step jitter would accumulate delay (~1s × steps)
                    # for no additional benefit, so we jitter only here.
                    #
                    # JITTER_S = 26s (measured inter-peak gap in rollout 0):
                    # uniformly scatters 256 trajectories across one full average
                    # step length, turning the GPU concurrency pulse into a
                    # plateau. Override with GUI_RESET_JITTER_S.
                    jitter_s = float(os.getenv("GUI_RESET_JITTER_S", "26"))
                    sleep_s = wait_after_reset + (random.uniform(0, jitter_s) if jitter_s > 0 else 0.0)
                    _gui_log("wait_after_reset sample=%s seconds=%.1f (base=%.1f jitter=%.1f)",
                             sample.index, sleep_s, wait_after_reset, jitter_s)
                    await asyncio.sleep(sleep_s)
                    obs = await env_client.get_obs(lease_id)
            if save_media:
                await asyncio.to_thread(_save_image, obs["screenshot"], result_dir / "step_0.png")
            _gui_log("lease ready sample=%s lease_id=%s", sample.index, lease_id)

            # Repetitive-action early stop: break when the same primary action
            # string repeats GUI_REPEAT_ACTION_THRESHOLD times in a row. Cuts
            # degenerate backspace/click loops that burn ~40min of wall-clock per
            # stuck trajectory without adding learning signal (reward, when
            # present, is earned before the loop starts). Env scores naturally
            # on the truncated state; reward is NOT forced to 0.
            repeat_threshold = int(os.getenv("GUI_REPEAT_ACTION_THRESHOLD", "5"))
            consecutive_repeats = 0
            last_action_key: str | None = None

            for step_idx in range(max_steps):
                timing.start_step(step_idx)
                # Check abort at the start of each step: if the rollout manager
                # killed this episode (e.g. weight update, dynamic filter, manual
                # stop), break immediately instead of burning env/judge resources.
                if state.aborted:
                    _gui_log("episode aborted sample=%s step=%s -> breaking", sample.index, step_idx)
                    break
                rollout_phase = "env_heartbeat"
                await env_client.heartbeat(lease_id)
                # Pure-CPU (PIL decode + smart_resize + PNG encode + message dicts):
                # run in a worker thread so 64 concurrent episodes don't serialize
                # the event loop behind it. Safe — touches only this sample's agent.
                rollout_phase = "policy_prepare"
                _t_prep = time.perf_counter()
                parse_ctx = await asyncio.to_thread(
                    parser.build_policy_messages, instruction=instruction, obs=obs
                )
                _prep_s = time.perf_counter() - _t_prep
                policy_messages = parse_ctx["messages"]
                tool_spec = parse_ctx.get("tool_spec")

                # Match feasible eval (mm_agents._infer @retry stop_after_attempt(7)):
                # empty / illegal tool-call parse is retried with a new sampling seed
                # before failing the trajectory. Do NOT record failed attempts into
                # history / step_snapshots — only the accepted parse is kept.
                parse_max_retries = int(
                    getattr(args, "gui_parse_max_retries", None)
                    or os.getenv("GUI_PARSE_MAX_RETRIES", "7")
                )
                parse_max_retries = max(1, parse_max_retries)
                response = ""
                finish_type = "abort"
                step_train_data: dict[str, Any] | None = None
                natural_action = ""
                actions: list[str] = []
                info_dict: dict[str, Any] = {}
                parse_ok = False
                for parse_attempt in range(parse_max_retries):
                    rollout_phase = "policy_generate"
                    with timing.step_generate():
                        response, finish_type, step_train_data = await parser.generate_with_sglang(
                            args=args,
                            state=state,
                            messages=policy_messages,
                            sampling_params=sampling_params,
                            sampling_seed=(
                                (int(sample.index or 0) + 1) * 1000003
                                + step_idx * 9973
                                + parse_attempt * 131
                            ),
                            tool_spec=tool_spec,
                            routing_key=episode_id,
                        )
                    preview_chars = int(os.getenv("GUI_LOG_RESPONSE_PREVIEW_CHARS", "0"))
                    rendered_response = response
                    if preview_chars > 0 and len(response) > preview_chars:
                        rendered_response = response[:preview_chars] + "...(truncated)"
                    _gui_log(
                        "step sample=%s step=%s attempt=%s/%s finish=%s response=%s",
                        sample.index,
                        step_idx,
                        parse_attempt + 1,
                        parse_max_retries,
                        finish_type,
                        rendered_response,
                    )
                    if finish_type in {"abort", "context_exhausted"}:
                        break

                    original_width = int(parse_ctx["original_width"])
                    original_height = int(parse_ctx["original_height"])
                    processed_width = int(parse_ctx["processed_width"])
                    processed_height = int(parse_ctx["processed_height"])

                    rollout_phase = "policy_parse"
                    natural_action, actions, info_dict = parser.parse_response(
                        response=response,
                        original_width=original_width,
                        original_height=original_height,
                        processed_width=processed_width,
                        processed_height=processed_height,
                    )
                    if actions and actions[0] != "":
                        parse_ok = True
                        break
                    _gui_log(
                        "parse retry sample=%s step=%s attempt=%s/%s empty_or_illegal_actions",
                        sample.index,
                        step_idx,
                        parse_attempt + 1,
                        parse_max_retries,
                    )

                timing.note_step(
                    prepare_s=round(_prep_s, 4),
                    gen_timing=(step_train_data or {}).get("gen_timing"),
                    finish_type=finish_type,
                    parse_ok=parse_ok,
                    gen_tokens=int(step_train_data["response_length"]) if step_train_data else 0,
                )

                if finish_type == "abort":
                    final_status = Sample.Status.ABORTED
                    timing.end_step()
                    break
                if finish_type == "context_exhausted":
                    # The prompt (history + screenshots) left no room to
                    # generate. End the episode as TRUNCATED so the steps already
                    # taken are kept and scored, instead of aborting the trajectory.
                    _gui_log("context exhausted sample=%s step=%s; ending episode", sample.index, step_idx)
                    final_status = Sample.Status.TRUNCATED
                    timing.note_step(stop_reason="context_exhausted")
                    timing.end_step()
                    break
                if not parse_ok or step_train_data is None:
                    final_status = Sample.Status.FAILED
                    timing.end_step()
                    break
                # Repetitive-action early stop: compare the primary action string
                # (actions[0]) against the previous step's. A run of
                # repeat_threshold identical primary actions in a row means the
                # policy is stuck in a degenerate loop (verified on rollouts
                # 0-3: 390/1024 trajectories loop, all rewards earned pre-loop).
                # Break as TRUNCATED so steps already taken are kept and scored;
                # reward is NOT forced to 0.
                action_key = actions[0] if actions else ""
                if action_key and action_key == last_action_key:
                    consecutive_repeats += 1
                else:
                    consecutive_repeats = 1
                    last_action_key = action_key
                if repeat_threshold > 1 and consecutive_repeats >= repeat_threshold:
                    _gui_log(
                        "repetitive action stop sample=%s step=%s repeats=%s action=%s -> truncating",
                        sample.index,
                        step_idx,
                        consecutive_repeats,
                        action_key,
                    )
                    final_status = Sample.Status.TRUNCATED
                    timing.note_step(stop_reason="repetitive_action", repeated_action=action_key,
                                     consecutive_repeats=consecutive_repeats)
                    timing.end_step()
                    break
                # Token passthrough: train data was captured at generation time
                # (prompt_ids + engine response tokens). No re-render, no
                # step_loss_mask bookkeeping — history is prompt by position.
                last_step_train_data_for_loss = step_train_data
                assistant_responses.append(response)
                step_snapshots.append(
                    {
                        "step_idx": step_idx,
                        "response_text": response,
                        "train_data": step_train_data,
                    }
                )

                parser.record_policy_turn(
                    action_text=natural_action or "Execute action",
                    response=response,
                    screenshot_bytes=obs["screenshot"],
                    # Same screenshot build_policy_messages already resized+encoded
                    # this step — skip the redundant PNG re-encode.
                    processed_b64=parse_ctx.get("processed_image_b64"),
                )

                step_executed = False
                done = False
                chunks = action_chunks(actions, enabled=os.getenv("GUI_BATCH_NAVIGATION_KEYS", "0") == "1")
                timing.note_step(action_rpc_count=len(chunks), parsed_action_count=len(actions))
                for chunk in chunks:
                    action = chunk.command
                    for primitive in chunk.actions:
                        timing.note_step_action(primitive)
                    marker = _logical_marker(action)
                    if marker is not None:
                        # Runner-side fake execute (guirl actor_client / markers.py).
                        # Do NOT call env.step — that asserts and would terminate the
                        # VM episode, breaking the later judge probe on this lease.
                        rollout_phase = "env_step"
                        with timing.step_env_exec():
                            obs, done = await _handle_logical_marker(
                                env_client,
                                lease_id=lease_id,
                                marker=marker,
                                sleep_after_execution=sleep_after_execution,
                                current_obs=obs,
                            )
                        reward, info = 0.0, {"logical_marker": marker}
                        if marker == "DONE":
                            final_status = Sample.Status.COMPLETED
                        elif marker == "FAIL":
                            final_status = Sample.Status.FAILED
                        _gui_log(
                            "marker sample=%s step=%s marker=%s done=%s",
                            sample.index,
                            step_idx,
                            marker,
                            done,
                        )
                    else:
                        rollout_phase = "env_step"
                        with timing.step_env_exec():
                            obs, reward, done, info = await env_client.step(
                                lease_id=lease_id, action=action, sleep_after_execution=sleep_after_execution
                            )
                        _gui_log(
                            "action sample=%s step=%s action=%s done=%s",
                            sample.index,
                            step_idx,
                            action,
                            done,
                        )
                        if done:
                            final_status = Sample.Status.COMPLETED

                    step_executed = True
                    if save_media and obs and obs.get("screenshot"):
                        step_image_path = result_dir / f"step_{step_idx + 1}.png"
                        await asyncio.to_thread(_save_image, obs["screenshot"], step_image_path)
                        screenshot_file = step_image_path.name
                    else:
                        screenshot_file = None
                    await asyncio.to_thread(
                        _write_json,
                        traj_path,
                        {
                            "step_num": step_idx + 1,
                            "action": action,
                            "primitive_actions": list(chunk.actions),
                            "natural_language_action": info_dict.get("action"),
                            "response": response,
                            "done": done,
                            "info": info,
                            "screenshot_file": screenshot_file,
                        },
                        append=True,
                    )
                    if done:
                        break

                if save_media:
                    trace_records.append(
                        {
                            "messages": policy_messages,
                            "response": response,
                            "actions": actions,
                            "step_idx": step_idx,
                            "step_executed": step_executed,
                        }
                    )
                timing.note_step(done=done)
                timing.end_step()

                if done:
                    break
            else:
                final_status = Sample.Status.TRUNCATED

            # Keep the final observation for the judge: parser.screenshots only
            # holds the screenshots the policy saw BEFORE each action, so the
            # post-final-action state (the one that decides task success) must be
            # appended explicitly.
            try:
                if obs and obs.get("screenshot"):
                    final_screenshot_b64 = await asyncio.to_thread(process_image, obs["screenshot"])
            except Exception:
                logger.warning("Failed to encode final screenshot for judge", exc_info=True)

            # Do NOT env.step("FAIL") here: that is a logical marker and would set
            # OSWorldEnv._terminated, breaking the judge probe on this same lease.
            # Non-COMPLETED status is already recorded on final_status above.

            # --------------------------------------------------------------
            # Rule-based env score FIRST — monitoring only, never a training
            # reward. Must run BEFORE the judge/probe loop: the holo3 probe steps
            # the SAME lease to gather evidence, mutating the environment, so a
            # post-probe evaluate() would score the probe's navigation state
            # instead of the actor's clean final state and pollute both ``acc``
            # and ``judge_env_agree``.
            # --------------------------------------------------------------
            try:
                rollout_phase = "env_evaluate"
                with timing.phase("env_evaluate"):
                    eval_score = await env_client.evaluate(lease_id=lease_id)
            except Exception as exc:
                eval_score = -1.0
                rollout_error = str(exc)
                # In env-reward mode this score IS the reward, so the episode
                # must be re-run. With a judge/zerogui reward it is monitoring
                # only: keep the (soon judge-scored) trajectory, record acc=-1.
                retryable_env_failure = reward_source == "env"
                logger.warning("env evaluate failed for sample %s; recording eval_score=-1", sample.index, exc_info=True)
            _gui_log("rollout end sample=%s status=%s env_score=%.4f", sample.index, final_status.value, eval_score)

            # --------------------------------------------------------------
            # Judge + probe reward (frozen reward agent). Runs WHILE the lease is
            # still open so the holo3 probe can step the SAME env to gather evidence
            # for "uncertain" criteria before re-judging. Freeing the VM is
            # deferred to finally.
            # --------------------------------------------------------------
            if reward_source in {"judge", "zerogui"} and final_status != Sample.Status.ABORTED and step_snapshots:
                rollout_phase = "judge"
                judge_screenshots = list(parser.screenshots)
                if final_screenshot_b64:
                    judge_screenshots.append(final_screenshot_b64)
                # The live lease is only needed past this point by the holo3
                # probe (judge mode with gui_probe_enable). Otherwise hand the
                # VM and the trajectory slot back NOW: judging a 30-step episode
                # can take minutes, and with GUI_TRAJECTORY_CONCURRENCY == pool
                # size every env would otherwise idle for that long.
                if reward_source == "zerogui" or not _probe_enabled(args):
                    rollout_phase = "env_close"
                    try:
                        await _close_lease_bounded(env_client, lease_id, timing)
                    except Exception:
                        logger.exception("Failed to close env lease early: %s", lease_id)
                    else:
                        lease_id = None
                    if not semaphore_released:
                        trajectory_semaphore.release()
                        semaphore_released = True
                    rollout_phase = "judge"
                with timing.phase("judge_probe_total"):
                    if reward_source == "zerogui":
                        # Passive ZeroGUI judge: scores the finished trajectory
                        # in one pass (no probe, no live lease). Separate module,
                        # never touches the reward-agent judge stack.
                        judge_result = await _run_zerogui_judge(
                            args=args,
                            instruction=instruction,
                            screenshots_b64=judge_screenshots,
                            action_history=list(parser.actions),
                            policy_model=str(getattr(args, "hf_checkpoint", "gui-policy-model")),
                            domain=domain,
                            example_id=example_id,
                        )
                    else:
                        judge_result = await _run_judge_with_probe(
                            args=args,
                            judge_client=judge_client,
                            result_dir=result_dir,
                            lease_id=lease_id,
                            fence_token=env_client.fence_token(lease_id),
                            instruction=instruction,
                            screenshots_b64=judge_screenshots,
                            action_history=list(parser.actions),
                            timing_records=timing.judge_rounds,
                        )

            rollout_phase = "result_write"
            await asyncio.to_thread((result_dir / "result.txt").write_text, f"{eval_score}\n", encoding="utf-8")
            if save_media:
                await asyncio.to_thread(
                    _write_json,
                    result_dir / "trajectory.json",
                    {"meta": {"result": eval_score}, "trajectory": trace_records},
                    append=False,
                )

            if judge_result is not None:
                _gui_log(
                    "judge verdict sample=%s status=%s rounds=%s probe=%s env_score=%.2f",
                    sample.index,
                    judge_result.status,
                    judge_result.num_rounds,
                    judge_result.used_probe,
                    eval_score,
                )
                await asyncio.to_thread(
                    _write_json,
                    result_dir / "judge.json",
                    {
                        "status": judge_result.status,
                        "criteria": judge_result.criteria_text,
                        "raw_text": judge_result.raw_text,
                        "num_rounds": judge_result.num_rounds,
                        "used_probe": judge_result.used_probe,
                        "probe_steps": judge_result.probe_steps,
                        "error": judge_result.error,
                        "env_score": eval_score,
                    },
                    append=False,
                )
    except TimeoutError as e:
        # A TimeoutError lands here for two distinct reasons; discriminate via
        # Timeout.expired(): (a) OUR trajectory deadline fired (asyncio.timeout
        # converts the internal cancellation to TimeoutError at ``async with``
        # exit, so the lease close in ``finally`` runs un-cancelled), or
        # (b) some inner client (aiohttp/env rpc) timed out on its own — that
        # is an ordinary rollout error, handled like any other exception.
        if not (isinstance(trajectory_deadline, asyncio.Timeout) and trajectory_deadline.expired()):
            # Identical handling to the generic ``except Exception`` below —
            # this TimeoutError predates the trajectory deadline feature and
            # must keep flowing exactly as before.
            final_status = Sample.Status.ABORTED
            rollout_error = str(e)
            retryable_env_failure = retryable_env_failure or rollout_phase.startswith("env_")
            tb = traceback.format_exc()
            await asyncio.to_thread(_write_json, traj_path, {'Error': str(e), 'Traceback': tb}, append=True)
            print(
                f"[GUI_ROLLOUT_ERROR] sample_index={sample.index} domain={domain} example_id={example_id}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
            logger.exception("GUI rollout failed for sample %s", sample.index)
        else:
            # Long-tail episode: abort it (-> remove_sample=True below), free
            # the env in ``finally`` and let the rollout proceed without it.
            timed_out = True
            final_status = Sample.Status.ABORTED
            # TimeoutError carries no message; keep a readable one for timing logs.
            rollout_error = f"trajectory timeout after {trajectory_timeout_s:.0f}s"
            await asyncio.to_thread(_write_json, traj_path, {'Error': f'trajectory timeout after {trajectory_timeout_s:.0f}s'}, append=True)
            _gui_log(
                "TRAJECTORY TIMEOUT sample=%s group=%s domain=%s example=%s timeout=%.0fs -> dropped from training, env released",
                sample.index,
                sample.group_index,
                domain,
                example_id,
                trajectory_timeout_s,
            )
            logger.warning(
                "GUI rollout timed out after %.0fs for sample %s (%s/%s); sample dropped, lease released",
                trajectory_timeout_s,
                sample.index,
                domain,
                example_id,
            )
    except Exception as e:
        from gui_actor_rl.dapo_http import InflightRequestUncertainError
        if isinstance(e, InflightRequestUncertainError):
            raise
        final_status = Sample.Status.ABORTED
        rollout_error = str(e)
        retryable_env_failure = retryable_env_failure or rollout_phase.startswith("env_")
        tb = traceback.format_exc()
        await asyncio.to_thread(_write_json, traj_path, {'Error': str(e), 'Traceback': tb}, append=True)
        print(
            f"[GUI_ROLLOUT_ERROR] sample_index={sample.index} domain={domain} example_id={example_id}\n{tb}",
            file=sys.stderr,
            flush=True,
        )
        logger.exception("GUI rollout failed for sample %s", sample.index)
    finally:
        cleanup_error = None
        try:
            try:
                if lease_id is not None:
                    try:
                        await _close_lease_bounded(env_client, lease_id, timing)
                    except Exception as exc:
                        cleanup_error = exc
                        logger.exception("Failed to close env lease: %s", lease_id)
            finally:
                try:
                    await env_client.close_session()
                except Exception as exc:
                    cleanup_error = exc
                    logger.warning("env session cleanup failed", exc_info=True)
        finally:
            if not semaphore_released:
                trajectory_semaphore.release()
                semaphore_released = True
        if cleanup_error is not None and getattr(args, 'rollout_lookahead_groups', 0):
            from gui_actor_rl.dapo_http import record_boundary_failure
            record_boundary_failure(args, f'Environment cleanup failed for sample {sample.index}')
            raise RuntimeError('GUI lookahead could not verify environment cleanup') from cleanup_error

    # Infrastructure failures are not task outcomes. Re-run the same policy
    # sample from a fresh allocation instead of dropping it immediately. The
    # deterministic sampling seed intentionally stays unchanged, so this is a
    # resubmission of the same episode rather than another policy sample.
    max_episode_retries = 0
    if not evaluation:
        configured_retries = getattr(args, "gui_episode_retries", None)
        if configured_retries is None:
            configured_retries = os.getenv("GUI_EPISODE_RETRIES", "2")
        max_episode_retries = max(0, int(configured_retries))
    if (
        retryable_env_failure
        and not timed_out
        and _episode_attempt < max_episode_retries
    ):
        next_attempt = _episode_attempt + 1
        _gui_log(
            "episode retry sample=%s attempt=%s/%s phase=%s reason=%s",
            sample.index,
            next_attempt + 1,
            max_episode_retries + 1,
            rollout_phase,
            rollout_error,
        )
        # This directory belongs to one sample only. Remove partial screenshots
        # and trace files so the retried episode cannot inherit stale evidence.
        await asyncio.to_thread(shutil.rmtree, result_dir, ignore_errors=True)
        if _retry_history is not None:
            _retry_history.append({"attempt": _episode_attempt + 1, "phase": rollout_phase,
                                   "elapsed_s": time.monotonic() - attempt_start,
                                   "error": str(rollout_error)[:500]})
        return _RETRY_EPISODE

    # Base sample train data: token passthrough from the LAST generated step
    # (same data the step snapshot carries). No re-render/re-tokenize.
    if last_step_train_data_for_loss is not None:
        input_ids = last_step_train_data_for_loss["tokens"]
        loss_mask = list(last_step_train_data_for_loss["loss_mask"])
        response_length = int(last_step_train_data_for_loss["response_length"])
        mm_train = last_step_train_data_for_loss["mm_train"]
    else:
        # No step completed (env allocate/reset failure, abort before the
        # first response). The sample is dropped from training below
        # (remove_sample=True), but slime does NOT physically remove dropped
        # samples — they still flow through get_batch, whose loss-mask
        # padding is F.pad(mask, (prompt_length - 1, 1)) and crashes with a
        # negative pad when tokens is empty (prompt_length == 0). Give the
        # sample a single sentinel prompt token so prompt_length == 1;
        # loss_mask stays [] (== response_length 0) and carries no gradient.
        sentinel = state.tokenizer.eos_token_id
        if sentinel is None:
            sentinel = 0
        input_ids = [int(sentinel)]
        loss_mask = []
        response_length = 0
        mm_train = None

    sample.tokens = input_ids
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.response_length = response_length
    sample.multimodal_train_inputs = mm_train
    sample.prefix_cache_info = _prefix_cache_info_of(last_step_train_data_for_loss)
    sample.rollout_log_probs = _rollout_log_probs_of(last_step_train_data_for_loss)
    sample.status = final_status
    sample.metadata = sample.metadata or {}
    sample.metadata["gui_result_dir"] = str(result_dir)
    sample.metadata["gui_score"] = eval_score
    sample.metadata["gui_episode_attempts"] = _episode_attempt + 1
    sample.metadata["gui_episode_retries"] = _episode_attempt
    if timed_out:
        sample.metadata["gui_trajectory_timed_out"] = True
    if reward_source == "env":
        # Rule-based reward straight from OSWorld's task evaluate() script
        # (ported from upstream gui_reward_source=env). score == acc by
        # construction; keep the judge_* keys (neutral values) so the reward
        # dict schema matches judge/zerogui mode for logging.
        env_ok = final_status != Sample.Status.ABORTED and step_snapshots and eval_score >= 0
        score = float(eval_score) if env_ok else 0.0
        sample.reward = {
            "score": score,
            "acc": max(float(eval_score), 0.0),
            "judge_env_agree": 1.0,
            "judge_used_probe": 0.0,
        }
        if not env_ok and not evaluation:
            # Aborted / empty trajectory / evaluate() error: no usable signal.
            sample.remove_sample = True
            _gui_log(
                "sample dropped from training: sample=%s env_reward status=%s eval_score=%.2f",
                sample.index,
                final_status.value,
                eval_score,
            )
    elif judge_result is not None:
        sample.metadata["judge"] = {
            "status": judge_result.status,
            "criteria": judge_result.criteria_text,
            "num_rounds": judge_result.num_rounds,
            "used_probe": judge_result.used_probe,
            "probe_steps": judge_result.probe_steps,
            "error": judge_result.error,
            # env evaluate() raised; acc/judge_env_agree are not meaningful here.
            "env_eval_failed": float(eval_score) < 0.0,
        }
        reward_dict, drop = _judge_reward(judge_result, eval_score)
        sample.reward = reward_dict
        if drop and not evaluation:
            sample.remove_sample = True
            _gui_log(
                "sample dropped from training: sample=%s judge_status=%s",
                sample.index,
                judge_result.status,
            )
    else:
        # No judge verdict (aborted rollout / empty trajectory): no signal.
        sample.reward = {"score": 0.0, "acc": float(eval_score), "judge_env_agree": 0.0, "judge_used_probe": 0.0}
        if not evaluation:
            sample.remove_sample = True

    # Timing/diagnostic record — AFTER the reward + drop decision so the record
    # carries the final training signal (no-op unless gui_timing_log_enable).
    try:
        await timing.finalize_async(
            status=final_status.value if hasattr(final_status, "value") else str(final_status),
            eval_score=eval_score,
            judge_status=judge_result.status if judge_result is not None else None,
            judge_used_probe=judge_result.used_probe if judge_result is not None else None,
            reward_score=float(sample.reward.get("score", 0.0)) if isinstance(sample.reward, dict) else None,
            removed_from_training=bool(sample.remove_sample or final_status == Sample.Status.ABORTED),
            result_dir=str(result_dir),
            error=rollout_error,
            retry_attempts=_retry_history,
        )
    except Exception:
        logger.warning("timing finalize failed", exc_info=True)

    if getattr(args, "dynamic_history", False) and not evaluation:
        # Any list[Sample] returned to slime must carry group_id on every
        # sibling (see _validate_group_id_annotated in slime/ray/rollout.py).
        sample.group_id = sample.index
        dynamic_samples = _build_dynamic_history_samples(
            args=args,
            state=state,
            agent=parser,
            base_sample=sample,
            step_snapshots=step_snapshots,
            outcome_reward=sample.reward,
        )
        result = dynamic_samples if dynamic_samples else [sample]
        _mark_aborted_samples(result)
        return result
    _mark_aborted_samples([sample])
    return sample


def _mark_aborted_samples(samples: list[Sample]) -> None:
    """Give ABORTED samples a default reward and exclude them from training."""
    for s in samples:
        if s.status == Sample.Status.ABORTED:
            if s.reward is None:
                s.reward = {"score": 0.0, "acc": 0.0, "judge_env_agree": 0.0, "judge_used_probe": 0.0}
            s.remove_sample = True


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    """--custom-rm-path hook: the reward is already computed by the judge in
    generate(); just pass it through."""

    def _single(s: Sample) -> dict[str, float]:
        if isinstance(s.reward, dict):
            return s.reward
        acc = float((s.metadata or {}).get("gui_score", 0.0))
        return {"score": 0.0, "acc": acc, "judge_env_agree": 0.0, "judge_used_probe": 0.0}

    if isinstance(sample, list):
        return [_single(s) for s in sample]
    return _single(sample)


def reward_post_process(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """--custom-reward-post-process-path hook: GRPO group normalization that is
    correct under dynamic_history fan-out and judge-dropped samples.

    Fixes two silent defects of slime's default ``_post_process_rewards``:

    1. dynamic_history returns a variable number of per-step child samples per
       trajectory, so the flat sample count never equals
       ``n_samples_per_prompt * rollout_batch_size`` and the default reshape
       falls back to one whole-batch group: the mean/std mixes every task and
       weights each trajectory by its step count. Here the group is
       ``sample.group_index`` (the actual n-samples-per-prompt group) and the
       baseline is computed over unique TRAJECTORIES (``sample.index``), so a
       30-step trajectory counts once, same as a 3-step one.

    2. Dropped samples (judge uncertain / parse_failure / aborted;
       ``remove_sample=True``, score=0.0) contribute nothing to the gradient
       (slime zeroes their loss_mask) but the default still lets their 0.0
       reward drag the group mean. Here they are excluded from the baseline;
       their own advantage is forced to 0.0 (harmless — loss is masked anyway).

    Returns ``(raw_rewards, rewards)`` aligned with ``samples`` order, matching
    the default hook contract in slime/ray/rollout.py::_post_process_rewards.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    # ── collect one reward per trajectory per group ──────────────────
    # group key: group_index (all rollouts of one prompt); trajectory key:
    # index (dynamic_history children copy index from their base sample).
    group_traj_reward: dict[Any, dict[Any, float]] = {}
    group_traj_valid: dict[Any, dict[Any, bool]] = {}
    for sample, raw in zip(samples, raw_rewards, strict=True):
        gkey = sample.group_index if sample.group_index is not None else sample.index
        tkey = sample.index
        group_traj_reward.setdefault(gkey, {})[tkey] = float(raw)
        # A trajectory is valid only if none of its samples were dropped.
        valid = not sample.remove_sample
        prev = group_traj_valid.setdefault(gkey, {})
        prev[tkey] = prev.get(tkey, True) and valid

    use_std = args.advantage_estimator in ["grpo", "gspo"] and getattr(args, "grpo_std_normalization", True)

    group_stats: dict[Any, tuple[float, float]] = {}
    for gkey, traj_rewards in group_traj_reward.items():
        valid_rewards = [r for tkey, r in traj_rewards.items() if group_traj_valid[gkey][tkey]]
        if not valid_rewards:
            group_stats[gkey] = (0.0, 1.0)
            continue
        mean = sum(valid_rewards) / len(valid_rewards)
        if use_std and len(valid_rewards) > 1:
            var = sum((r - mean) ** 2 for r in valid_rewards) / (len(valid_rewards) - 1)
            std = var**0.5
        else:
            std = 0.0
        group_stats[gkey] = (mean, std)

    rewards: list[float] = []
    for sample, raw in zip(samples, raw_rewards, strict=True):
        gkey = sample.group_index if sample.group_index is not None else sample.index
        if sample.remove_sample or not group_traj_valid[gkey].get(sample.index, False):
            rewards.append(0.0)
            continue
        mean, std = group_stats[gkey]
        adv = float(raw) - mean
        if use_std:
            adv = adv / (std + 1e-6)
        rewards.append(adv)

    return raw_rewards, rewards


def _episode_metrics(samples, prefix):
    episodes = {}

    def flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    for sample in flatten(samples):
        key = (sample.group_index, sample.index)
        reward = float((sample.metadata or {}).get("gui_score", -1))
        valid = not sample.remove_sample and sample.status != Sample.Status.ABORTED and reward >= 0
        previous = episodes.get(key)
        episodes[key] = (reward, valid and (previous is None or previous[1]))
    valid_rewards = [reward for reward, valid in episodes.values() if valid]
    groups = {}
    for (group, _), (reward, valid) in episodes.items():
        if valid:
            groups.setdefault(group, []).append(reward)
    return {
        prefix + "episodes": len(episodes),
        prefix + "valid_episodes": len(valid_rewards),
        prefix + "failed_episodes": len(episodes) - len(valid_rewards),
        prefix + "episode_reward_mean": sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0,
        prefix + "episode_full_success_rate": sum(r >= 1 - 1e-6 for r in valid_rewards) / len(valid_rewards) if valid_rewards else 0.0,
        prefix + "reward_varying_group_fraction": sum(max(rs) > min(rs) for rs in groups.values()) / len(groups) if groups else 0.0,
    }


def _require_complete_gui_batch(metrics, expected):
    if metrics["rollout/gui/episodes"] != expected or metrics["rollout/gui/valid_episodes"] != expected:
        raise RuntimeError(f"Incomplete GUI training batch: expected {expected} valid trajectories, "
                           f"got {metrics['rollout/gui/valid_episodes']}; refusing a partial training update. "
                           "See rollout timing summary for failed task IDs.")


def gui_generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    # Stamp the rollout id so per-sample timing records (gui_timing_log_enable)
    # know which training rollout/task-batch they belong to. Requires this
    # function to be wired as BOTH --rollout-function-path and
    # --eval-function-path (see scripts/run_gui_actor_rl_qwen3vl_4b.sh).
    set_current_rollout_id(rollout_id)

    # Train path keeps the default rollout behavior.
    if not evaluation:
        from slime.rollout.sglang_rollout import generate_rollout
        from slime.utils.async_utils import run
        from gui_actor_rl.timing import start_event_loop_monitor, timing_enabled

        loop_metrics = {}
        dapo_audit = None
        if os.getenv("GUI_DAPO", "0") == "1":
            from gui_actor_rl.dapo import configure, write_audit
            dapo_audit = configure(args)
        monitor = run(start_event_loop_monitor()) if timing_enabled(args) else None
        try:
            if dapo_audit is not None and args.rollout_lookahead_groups:
                from gui_actor_rl.dapo_scheduler import generate_lookahead
                output = run(generate_lookahead(args, rollout_id, data_source.get_samples))
            else:
                output = generate_rollout(args, rollout_id, data_source, evaluation=False)
        except BaseException as exc:
            if dapo_audit is not None:
                write_audit(args, rollout_id, dapo_audit, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if monitor is not None:
                loop_metrics = run(monitor.stop())
        output.metrics = {**(output.metrics or {}), **_episode_metrics(output.samples, "rollout/gui/"), **loop_metrics}
        if dapo_audit is not None:
            output.metrics.update(dapo_audit.metrics())
            write_audit(args, rollout_id, dapo_audit)
        write_rollout_summary(args, rollout_id)
        if os.getenv("GUI_REQUIRE_COMPLETE_BATCH", "0") == "1":
            _require_complete_gui_batch(output.metrics, args.rollout_batch_size * args.n_samples_per_prompt)
        return output

    from slime.utils.async_utils import run

    output, _ = run(_gui_eval_rollout(args))
    write_rollout_summary(args, rollout_id)
    return output


async def _gui_eval_rollout(args):
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import generate_and_rm

    cua_manifest = getattr(args, "gui_cua_gym_eval_manifest", "")
    if cua_manifest:
        from gui_actor_rl.cua_gym_data_source import CuaGymDataSource

        eval_args = copy.copy(args)
        eval_args.gui_cua_gym_manifest = cua_manifest
        tasks_to_eval = CuaGymDataSource(eval_args).tasks
    else:
        base_dir = os.getenv("GUI_TEST_CONFIG_BASE_DIR", _DEFAULT_CONFIG_BASE_DIR)
        meta_path = os.getenv("GUI_EVAL_META_PATH") or str(Path(base_dir) / _DEFAULT_EVAL_META)
        tasks_to_eval = []
        for domain, example_id in _load_meta_pairs(meta_path):
            cfg_path = Path(base_dir) / "examples" / str(domain) / f"{example_id}.json"
            if cfg_path.exists():
                config = _load_task_config(base_dir, domain, example_id)
                tasks_to_eval.append({"domain": domain, "example_id": example_id,
                                      "task_config": config, "instruction": config.get("instruction", "")})
    if not tasks_to_eval:
        raise RuntimeError("No eval tasks loaded")

    eval_max_response_len = getattr(args, "eval_max_response_len", None)
    if eval_max_response_len is None:
        eval_max_response_len = getattr(args, "rollout_max_response_len", 512)

    eval_top_p = getattr(args, "eval_top_p", None)
    if eval_top_p is None:
        eval_top_p = getattr(args, "rollout_top_p", 1.0)
    eval_top_k = getattr(args, "eval_top_k", None)
    if eval_top_k is None:
        eval_top_k = getattr(args, "rollout_top_k", -1)

    sampling_params = dict(
        temperature=float(getattr(args, "eval_temperature", 0.0) or 0.0),
        top_p=float(eval_top_p),
        top_k=int(eval_top_k),
        max_new_tokens=int(eval_max_response_len),
        stop=getattr(args, "rollout_stop", None),
        stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
        skip_special_tokens=getattr(args, "rollout_skip_special_tokens", False),
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    n_samples = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    tasks = []
    sample_index = 0
    for task in tasks_to_eval:
        domain, example_id = task["domain"], task["example_id"]
        task_config = task["task_config"]
        instruction = str(task["instruction"])
        group_index = sample_index // n_samples
        for _ in range(n_samples):
            sample = Sample(
                prompt=instruction,
                label="",
                metadata={
                    "domain": domain,
                    "example_id": example_id,
                    "instruction": instruction,
                    "task_config": task_config,
                },
            )
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            tasks.append(
                asyncio.create_task(generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True))
            )

    if not tasks:
        raise RuntimeError(f"No valid eval tasks found from {meta_path}")

    data = []
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)

    data.sort(key=lambda s: s.index)
    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", "score")
    rewards = []
    for sample in data:
        if isinstance(sample.reward, dict):
            rewards.append(float(sample.reward.get(reward_key, 0.0)))
        elif sample.reward is not None:
            rewards.append(float(sample.reward))
        else:
            rewards.append(0.0)

    invalid = [sample.index for sample in data
               if sample.status == Sample.Status.ABORTED
               or float((sample.metadata or {}).get("gui_score", -1)) < 0]
    if invalid:
        raise RuntimeError(f"Evaluation infrastructure failures in {len(invalid)} samples: {invalid[:16]}")

    return RolloutFnEvalOutput(
        metrics=_episode_metrics(data, "eval/gui/"),
        data={
            "gui_eval": {
                "rewards": rewards,
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        }
    ), []
