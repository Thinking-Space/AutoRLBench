"""Frozen-judge inference for GUI actor RL — the judge->probe->judge state machine.

This replicates, byte-for-byte in behavior, the judge-scoring loop from
``guirl.slime_custom.custom_generate`` (roughly lines 560–810) — the SAME loop
used while the judge was being TRAINED — with all *training* mechanics dropped
(token accumulation, loss_mask, logprobs, fan-out, multimodal encoding). The
judge here is FROZEN and served on a standalone sglang ``/chat/completions``, so
it produces no gradients; only its verdict matters.

What is aligned to training (do not change without changing slime_custom too):
- prompt = ``[{system: JUDGE_SYSTEM_PROMPT}, {user: build_judge_user_content(...)}]``;
- linear messages list, appended each round (NOT rebuilt every round);
- ``for round_idx in range(max_rounds)`` with the exact stop-condition ORDER;
- history carries only conclusions (``_strip_thinking_for_history``);
- probe on "uncertain" via the rollout master (holo3 harness), whose
  ``latest_criteria`` + ``criteria_images`` feed the next round DIRECTLY;
- parse retry keeps temperature (=1.0); exhaust -> "parse_failure";
- sampling temperature=1.0 / max_tokens=24576 / enable_thinking=true.

Only two things differ from guirl (because we train the actor, not the judge):
reward mapping is actor-semantics (handled by the caller), and the judge is a
frozen HTTP endpoint instead of the training engine.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from guirl.judge.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    JudgeStatus,
    _strip_thinking_for_history,
    build_judge_user_content,
)

from gui_actor_rl.judge_client import JudgeClient, JudgeResult
from gui_actor_rl.probe_client import RolloutMasterProbeRunner

logger = logging.getLogger(__name__)


async def run_judge_inference(
    *,
    judge_client: JudgeClient,
    probe_runner: RolloutMasterProbeRunner | None,
    lease_id: str | None,
    fence_token: str | None = None,
    instruction: str,
    trajectory_screenshots: list[str],
    action_history: list[str] | None,
    max_rounds: int = 3,
    history_n: int = 3,
    timing_records: list[dict[str, Any]] | None = None,
) -> JudgeResult:
    """Run the frozen judge's judge->probe->judge loop over one actor trajectory.

    Mirrors ``guirl.slime_custom.custom_generate``'s judge loop (inference only).
    Never raises: probe/network/parse failures degrade to a usable verdict or to
    ``status="parse_failure"`` so the caller can drop the sample from training.

    Must be called while the env ``lease_id`` is still open — the probe steps the
    SAME VM to gather evidence for "uncertain" criteria.

    ``timing_records`` (optional, gated instrumentation): when provided, one dict
    per round is appended: ``{round, judge_s, judge_attempts, judge_status,
    probe_s, probe_steps}``. Pure bookkeeping — never changes behavior.
    """
    if not trajectory_screenshots:
        return JudgeResult(status="parse_failure", error="empty trajectory (no screenshots)")

    # === Unified linear messages list (mirrors slime_custom / evaluation_harness). ===
    # Round 0 user turn: trajectory screenshots + action history, no prev criteria.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
    ]
    user_content_r0 = build_judge_user_content(
        task=instruction,
        task_traj_imgs=trajectory_screenshots,
        prev_criteria=None,
        action_history=list(action_history) if action_history else None,
        history_n=history_n,
        criteria_option_screenshot=None,
    )
    messages.append({"role": "user", "content": user_content_r0})

    current_status: JudgeStatus = "uncertain"
    current_criteria: str = ""
    last_response_text: str = ""
    num_rounds = 0
    used_probe = False
    probe_steps = 0

    for round_idx in range(max_rounds):
        num_rounds = round_idx + 1
        round_timing: dict[str, Any] | None = None
        if timing_records is not None:
            round_timing = {"round": round_idx, "judge_s": 0.0, "judge_attempts": 0,
                            "judge_status": "", "probe_s": 0.0, "probe_steps": 0}
            timing_records.append(round_timing)

        # --- judge generate with parse-retry (keeps temperature; -> parse_failure). ---
        _judge_t0 = time.monotonic()
        response_text, current_status, current_criteria, _attempts = await judge_client.chat_parseable(
            messages, round_idx=round_idx
        )
        if round_timing is not None:
            round_timing["judge_s"] = round(time.monotonic() - _judge_t0, 3)
            round_timing["judge_attempts"] = _attempts
            round_timing["judge_status"] = current_status
        last_response_text = response_text

        # --- Stop conditions, in the SAME ORDER as slime_custom. ---
        is_last_round = (
            current_status == "parse_failure"
            or current_status != "uncertain"
            or round_idx >= max_rounds - 1
            or lease_id is None
        )

        # Append this round's assistant reply to history BEFORE deciding to stop,
        # with thinking stripped so the next round sees only conclusions (aligned
        # to slime_custom, which also strips history thinking on continuing rounds).
        # On parse_failure there is nothing parseable to carry; skip the append.
        if current_status != "parse_failure":
            messages.append(
                {"role": "assistant", "content": _strip_thinking_for_history(response_text)}
            )

        if is_last_round:
            break

        # --- uncertain + not last round + lease alive + probe available -> probe. ---
        if probe_runner is None:
            # No probe configured: cannot resolve "uncertain"; stop here and let
            # the caller drop the sample (actor-semantics).
            break

        try:
            _probe_t0 = time.monotonic()
            probe_result = await probe_runner.run_probe_round(
                lease_id=lease_id,
                fence_token=fence_token,
                criteria_text=current_criteria,
                task_instruction=instruction,
            )
            if round_timing is not None:
                round_timing["probe_s"] = round(time.monotonic() - _probe_t0, 3)
                round_timing["probe_steps"] = int(probe_result.num_steps)
        except Exception as exc:
            if round_timing is not None:
                round_timing["probe_s"] = round(time.monotonic() - _probe_t0, 3)
            logger.warning("probe round failed (round %d): %s", round_idx, exc, exc_info=True)
            break

        used_probe = True
        probe_steps += int(probe_result.num_steps)

        # Update criteria from the probe (aligned: probe_result["latest_criteria"]),
        # and feed the probe's evidence screenshots into the next judge round
        # (aligned: probe_result["criteria_images"] as criteria_option_screenshot).
        current_criteria = probe_result.latest_criteria or current_criteria
        evidence = probe_result.criteria_images or None
        user_content_rN = build_judge_user_content(
            task=instruction,
            task_traj_imgs=trajectory_screenshots,
            prev_criteria=current_criteria,
            action_history=None,
            history_n=history_n,
            criteria_option_screenshot=evidence,
        )
        messages.append({"role": "user", "content": user_content_rN})

    return JudgeResult(
        status=current_status,
        criteria_text=current_criteria,
        raw_text=last_response_text,
        attempts=num_rounds,
        num_rounds=num_rounds,
        used_probe=used_probe,
        probe_steps=probe_steps,
    )
