"""Gated rollout-side timing instrumentation for GUI actor RL.

Purpose: diagnose why rollout wall-clock grows across training (e.g. 40min ->
100min per rollout) by recording, PER TRAJECTORY, how long every phase takes:

    semaphore wait -> env allocate -> env reset -> actor step 0..N
    (generation vs env execution) -> env evaluate -> judge round 0
    -> probe round 0 -> judge round 1 -> ... -> env close

together with WHICH task the trajectory belongs to (rollout_id, sample index,
group index, domain, example_id, instruction) and the final outcome
(status, env score, judge verdict).

Everything is gated by ``gui_timing_log_enable`` (runtime YAML, injected via
--custom-config-path) or the env var ``GUI_TIMING_LOG_ENABLE``. When disabled
every recorder method is a no-op and nothing is written.

Output layout (easy to eyeball, one dir per rollout step)::

    ${GUI_TIMING_LOG_DIR}/                      # default: <GUI_RESULT_DIR>/../logs/rollout_timing
        rollout_0003/
            chrome__abc-uuid__s17.json          # full per-trajectory detail
        rollout_0003.jsonl                      # one line per finished trajectory (streamed)
        rollout_0003_summary.json               # aggregate written when the rollout ends

The recorder is purely wall-clock bookkeeping around existing awaits — it never
changes rollout/judge behavior.
"""
from __future__ import annotations

import json
import asyncio
import logging
import os
import re
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_TIMING_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-timing-writer")


class EventLoopLagMonitor:
    """Bounded counters; one monitor per rollout, including failed rollouts."""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.count = self.over_100ms = 0
        self.total = self.maximum = 0.0
        self.task = asyncio.create_task(self._sample())

    async def _sample(self):
        while True:
            start = time.monotonic()
            await asyncio.sleep(self.interval)
            lag = max(0.0, time.monotonic() - start - self.interval)
            self.count += 1
            self.total += lag
            self.maximum = max(self.maximum, lag)
            self.over_100ms += lag >= 0.1

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        return {"perf/event_loop_lag_mean_s": self.total / max(1, self.count),
                "perf/event_loop_lag_max_s": self.maximum,
                "perf/event_loop_lag_over_100ms": self.over_100ms,
                "perf/event_loop_lag_samples": self.count}


async def start_event_loop_monitor():
    return EventLoopLagMonitor()

# Set by gui_generate_rollout (the --rollout-function-path /
# --eval-function-path wrapper) before samples are generated, so per-sample
# recorders know which rollout step they belong to.
_CURRENT_ROLLOUT_ID: int = -1


def set_current_rollout_id(rollout_id: int) -> None:
    global _CURRENT_ROLLOUT_ID
    _CURRENT_ROLLOUT_ID = int(rollout_id)


def get_current_rollout_id() -> int:
    return _CURRENT_ROLLOUT_ID


def timing_enabled(args: Any) -> bool:
    cfg = getattr(args, "gui_timing_log_enable", None)
    if cfg is not None:
        return bool(cfg)
    return os.getenv("GUI_TIMING_LOG_ENABLE", "0").strip().lower() in {"1", "true", "yes"}


def timing_log_dir(args: Any) -> Path:
    configured = getattr(args, "gui_timing_log_dir", None) or os.getenv("GUI_TIMING_LOG_DIR", "")
    if configured:
        return Path(str(configured))
    # GUI_RESULT_DIR defaults to <run_dir>/results -> put timing next to it
    # under <run_dir>/logs/rollout_timing so it sits with the other run logs.
    result_dir = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    return result_dir.parent / "logs" / "rollout_timing"


def _safe_tag(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", str(text))[:120]


class SampleTimingRecorder:
    """Per-trajectory wall-clock recorder. All methods no-op when disabled."""

    def __init__(self, args: Any, *, sample_index: Any, group_index: Any,
                 domain: str, example_id: str, instruction: str, evaluation: bool):
        self.enabled = timing_enabled(args)
        if not self.enabled:
            return
        self.base_dir = timing_log_dir(args)
        self.rollout_id = get_current_rollout_id()
        self.sample_index = sample_index
        self.record: dict[str, Any] = {
            "rollout_id": self.rollout_id,
            "sample_index": sample_index,
            "group_index": group_index,
            "domain": domain,
            "example_id": example_id,
            "instruction": instruction,
            "evaluation": evaluation,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "phases": {},          # phase name -> seconds (env_allocate, env_reset, ...)
            "steps": [],           # per actor step: {step, total_s, gen_s, gen_attempts, env_exec_s, num_actions}
            "judge_rounds": [],    # per judge round: {round, judge_s, judge_attempts, probe_s, probe_steps}
        }
        self._t0 = time.monotonic()

    # ── phases ────────────────────────────────────────────────────────
    @contextmanager
    def phase(self, name: str):
        """Time a named phase (env_allocate / env_reset / env_evaluate / ...)."""
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.record["phases"][name] = round(self.record["phases"].get(name, 0.0) + elapsed, 3)

    # ── actor steps ───────────────────────────────────────────────────
    def start_step(self, step_idx: int) -> None:
        if not self.enabled:
            return
        self._step = {
            "step": step_idx, "total_s": 0.0, "gen_s": 0.0,
            "gen_attempts": 0, "env_exec_s": 0.0, "num_actions": 0,
            # per-step diagnostics (why did this step go wrong?)
            "finish_type": "",     # sglang finish reason of the ACCEPTED attempt ("length" = truncated thinking)
            "gen_tokens": 0,       # response tokens of the accepted attempt
            "parse_ok": False,     # tool-call parse succeeded within retries
            "actions": [],         # executed action strings (truncated), incl. WAIT/DONE/FAIL markers
            "done": False,
        }
        self._step_t0 = time.monotonic()

    def note_step(self, **kwargs: Any) -> None:
        """Attach diagnostic key/values to the current step (no-op when disabled)."""
        if not self.enabled or not hasattr(self, "_step"):
            return
        self._step.update(kwargs)

    def note_step_action(self, action: Any) -> None:
        if not self.enabled or not hasattr(self, "_step"):
            return
        self._step["actions"].append(str(action)[:120])

    @contextmanager
    def step_generate(self):
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            self._step["gen_s"] += time.monotonic() - start
            self._step["gen_attempts"] += 1

    @contextmanager
    def step_env_exec(self):
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            self._step["env_exec_s"] += time.monotonic() - start
            self._step["num_actions"] += 1

    def end_step(self) -> None:
        if not self.enabled or not hasattr(self, "_step"):
            return
        self._step["total_s"] = round(time.monotonic() - self._step_t0, 3)
        self._step["gen_s"] = round(self._step["gen_s"], 3)
        self._step["env_exec_s"] = round(self._step["env_exec_s"], 3)
        self.record["steps"].append(self._step)
        del self._step

    # ── judge/probe rounds (filled by judge_infer via this list) ─────
    @property
    def judge_rounds(self) -> list[dict[str, Any]] | None:
        return self.record["judge_rounds"] if self.enabled else None

    # ── finalize ──────────────────────────────────────────────────────
    async def finalize_async(self, **kwargs) -> None:
        # Files live on shared storage. Even one slow mkdir/write used to
        # stall every socket/timeout in the rollout event loop. A single
        # writer preserves JSONL record ordering without concurrent append.
        if self.enabled:
            await asyncio.get_running_loop().run_in_executor(
                _TIMING_WRITER, partial(
                    self.finalize, **kwargs, _end_time=datetime.now().isoformat(timespec="seconds"),
                    _elapsed_s=round(time.monotonic() - self._t0, 3)
                )
            )

    def finalize(self, *, status: str, eval_score: float,
                 judge_status: str | None, judge_used_probe: bool | None,
                 result_dir: str | None = None,
                 reward_score: float | None = None,
                 removed_from_training: bool | None = None,
                 error: str | None = None, retry_attempts: list | None = None,
                 _end_time=None, _elapsed_s=None) -> None:
        """Write the per-sample JSON + append to the per-rollout JSONL."""
        if not self.enabled:
            return
        rec = self.record
        rec["end_time"] = _end_time or datetime.now().isoformat(timespec="seconds")
        rec["total_s"] = round(time.monotonic() - self._t0, 3) if _elapsed_s is None else _elapsed_s
        # Historical total_s is the final attempt only. Expose prior attempts
        # too, so a recovered reset does not hide minutes of rollout latency.
        rec["retry_attempts"] = list(retry_attempts or [])
        rec["retry_total_s"] = round(sum(r["elapsed_s"] for r in rec["retry_attempts"]), 3)
        rec["total_with_retries_s"] = rec["total_s"] + rec["retry_total_s"]
        rec["status"] = status
        rec["env_score"] = eval_score
        rec["judge_status"] = judge_status
        rec["judge_used_probe"] = judge_used_probe
        rec["reward_score"] = reward_score
        rec["removed_from_training"] = removed_from_training
        rec["error"] = (str(error)[:500] if error else None)
        rec["result_dir"] = result_dir
        rec["num_steps"] = len(rec["steps"])
        rec["gen_total_s"] = round(sum(s["gen_s"] for s in rec["steps"]), 3)
        rec["env_exec_total_s"] = round(sum(s["env_exec_s"] for s in rec["steps"]), 3)
        rec["judge_total_s"] = round(sum(r.get("judge_s", 0.0) for r in rec["judge_rounds"]), 3)
        rec["probe_total_s"] = round(sum(r.get("probe_s", 0.0) for r in rec["judge_rounds"]), 3)
        # trajectory-level diagnostics (each is a "why is acc low?" signal)
        steps = rec["steps"]
        rec["gen_attempts_total"] = sum(s.get("gen_attempts", 0) for s in steps)
        rec["parse_fail_steps"] = sum(1 for s in steps if not s.get("parse_ok", True))
        rec["truncated_steps"] = sum(1 for s in steps if s.get("finish_type") == "length")
        rec["gen_tokens_total"] = sum(s.get("gen_tokens", 0) for s in steps)
        rec["gen_tokens_max"] = max((s.get("gen_tokens", 0) for s in steps), default=0)
        rec["terminated_by"] = _terminated_by(steps, status)

        try:
            rollout_tag = f"rollout_{self.rollout_id:04d}" if self.rollout_id >= 0 else "rollout_unknown"
            detail_dir = self.base_dir / rollout_tag
            detail_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{_safe_tag(rec['domain'])}__{_safe_tag(rec['example_id'])}__s{rec['sample_index']}.json"
            with open(detail_dir / fname, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            with open(self.base_dir / f"{rollout_tag}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Global per-task history: one compact line per trajectory, ALL
            # rollouts in one file — grep a task id to see its whole training
            # history (env/judge score over time) at a glance.
            compact = {k: rec.get(k) for k in (
                "rollout_id", "sample_index", "group_index", "domain", "example_id",
                "status", "env_score", "judge_status", "reward_score",
                "removed_from_training", "judge_used_probe", "num_steps",
                "terminated_by", "truncated_steps", "parse_fail_steps",
                "gen_attempts_total", "gen_tokens_max", "total_s", "retry_total_s", "total_with_retries_s", "end_time", "error",
            )}
            with open(self.base_dir / "task_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(compact, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("failed to write timing record", exc_info=True)

        logger.info(
            "[GUI-TIMING] rollout=%s sample=%s %s/%s total=%.1fs "
            "(alloc=%.1fs reset=%.1fs gen=%.1fs env_exec=%.1fs eval=%.1fs judge=%.1fs probe=%.1fs close=%.1fs) "
            "steps=%d term=%s trunc=%d parse_fail=%d judge_rounds=%d status=%s judge=%s env_score=%.2f",
            self.rollout_id, rec["sample_index"], rec["domain"], rec["example_id"], rec["total_s"],
            rec["phases"].get("env_allocate", 0.0), rec["phases"].get("env_reset", 0.0),
            rec["gen_total_s"], rec["env_exec_total_s"], rec["phases"].get("env_evaluate", 0.0),
            rec["judge_total_s"], rec["probe_total_s"], rec["phases"].get("env_close", 0.0),
            rec["num_steps"], rec["terminated_by"], rec["truncated_steps"], rec["parse_fail_steps"],
            len(rec["judge_rounds"]), status, judge_status, eval_score,
        )


def _terminated_by(steps: list[dict[str, Any]], status: str) -> str:
    """How did the trajectory end? One of:
    done_marker (model said DONE/FAIL) / env_done / max_steps / repetitive_action / context_exhausted /
    parse_fail (retries exhausted) / abort / unknown."""
    if status == "aborted":
        return "abort"
    if steps:
        last = steps[-1]
        if last.get("stop_reason") in {"repetitive_action", "context_exhausted"}:
            return last["stop_reason"]
        if not last.get("parse_ok", True):
            return "parse_fail"
        actions = last.get("actions") or []
        if any(str(a).strip().upper() in {"DONE", "FAIL"} for a in actions):
            return "done_marker"
        if last.get("done"):
            return "env_done"
    if status == "truncated":
        return "max_steps"
    return status or "unknown"


def write_rollout_summary(args: Any, rollout_id: int) -> None:
    """Aggregate the per-rollout JSONL into rollout_XXXX_summary.json.

    Called by gui_generate_rollout after the rollout finishes. Reads the JSONL
    back so it works regardless of which coroutine wrote each record.
    """
    if not timing_enabled(args):
        return
    base_dir = timing_log_dir(args)
    rollout_tag = f"rollout_{rollout_id:04d}"
    jsonl_path = base_dir / f"{rollout_tag}.jsonl"
    if not jsonl_path.exists():
        return
    try:
        records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        logger.warning("failed to read timing jsonl %s", jsonl_path, exc_info=True)
        return
    if not records:
        return

    def _agg(key: str) -> dict[str, float]:
        vals = [float(r.get(key, 0.0) or 0.0) for r in records]
        return {
            "sum": round(sum(vals), 1),
            "mean": round(sum(vals) / len(vals), 1),
            "max": round(max(vals), 1),
        }

    # ── outcome metrics, PER TRAJECTORY (unweighted by dynamic_history fan-out,
    # unlike wandb rollout/acc which averages over child samples) ──
    # NOTE: env_score 0.0 is a legitimate failure score; only None / <0
    # (evaluate() errored, recorded as -1) is excluded. Do not use `or` here.
    def _env_score(r: dict[str, Any]) -> float:
        v = r.get("env_score")
        return -1.0 if v is None else float(v)

    scored = [r for r in records if _env_score(r) >= 0.0]
    env_acc = round(sum(_env_score(r) for r in scored) / len(scored), 4) if scored else None
    judge_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}
    for r in records:
        judge_counts[str(r.get("judge_status"))] = judge_counts.get(str(r.get("judge_status")), 0) + 1
        term_counts[str(r.get("terminated_by"))] = term_counts.get(str(r.get("terminated_by")), 0) + 1
    judge_success = judge_counts.get("success", 0)
    judge_failure = judge_counts.get("failure", 0)
    judge_decided = judge_success + judge_failure

    # ── per-task breakdown: which tasks dragged this rollout down ──
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_task.setdefault(f"{r.get('domain')}/{r.get('example_id')}", []).append(r)
    tasks = []
    for task_key, rs in sorted(by_task.items()):
        t_scored = [r for r in rs if _env_score(r) >= 0.0]
        tasks.append({
            "task": task_key,
            "n": len(rs),
            "env_acc": round(sum(_env_score(r) for r in t_scored) / len(t_scored), 3) if t_scored else None,
            "judge_success": sum(1 for r in rs if r.get("judge_status") == "success"),
            "dropped": sum(1 for r in rs if r.get("removed_from_training")),
            "truncated_steps": sum(int(r.get("truncated_steps", 0) or 0) for r in rs),
            "parse_fail_steps": sum(int(r.get("parse_fail_steps", 0) or 0) for r in rs),
            "terminated_by": sorted({str(r.get("terminated_by")) for r in rs}),
            "mean_steps": round(sum(int(r.get("num_steps", 0) or 0) for r in rs) / len(rs), 1),
        })

    slowest = sorted(records, key=lambda r: -float(r.get("total_s", 0.0) or 0.0))[:10]
    summary = {
        "rollout_id": rollout_id,
        "num_trajectories": len(records),
        # outcome (per-trajectory; compare directly against offline baseline acc)
        "env_acc_per_traj": env_acc,
        "judge_success_rate": round(judge_success / judge_decided, 4) if judge_decided else None,
        "judge_status_counts": judge_counts,
        "dropped_from_training": sum(1 for r in records if r.get("removed_from_training")),
        "terminated_by_counts": term_counts,
        # failure-mode signals ("why is acc low?")
        "trajs_with_truncated_steps": sum(1 for r in records if int(r.get("truncated_steps", 0) or 0) > 0),
        "truncated_steps_total": sum(int(r.get("truncated_steps", 0) or 0) for r in records),
        "parse_fail_steps_total": sum(int(r.get("parse_fail_steps", 0) or 0) for r in records),
        "gen_attempts_total": sum(int(r.get("gen_attempts_total", 0) or 0) for r in records),
        "gen_tokens_max": max((int(r.get("gen_tokens_max", 0) or 0) for r in records), default=0),
        "wall_clock": {k: _agg(k) for k in
                       ("total_s", "gen_total_s", "env_exec_total_s", "judge_total_s", "probe_total_s")},
        "phases": {name: _agg_phase(records, name) for name in
                   ("semaphore_wait", "env_allocate", "env_reset", "env_evaluate", "env_close")},
        "probe_fired_count": sum(1 for r in records if r.get("judge_used_probe")),
        "tasks": tasks,
        "slowest_trajectories": [
            {k: r.get(k) for k in ("sample_index", "domain", "example_id", "total_s",
                                   "gen_total_s", "env_exec_total_s", "judge_total_s",
                                   "probe_total_s", "num_steps", "status", "judge_status")}
            for r in slowest
        ],
    }
    try:
        with open(base_dir / f"{rollout_tag}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.warning("failed to write timing summary", exc_info=True)
    logger.info(
        "[GUI-TIMING] %s summary: n=%d env_acc=%s judge_sr=%s dropped=%d "
        "trunc_steps=%d parse_fail=%d term=%s | total(mean/max)=%.0f/%.0fs gen=%.0fs env=%.0fs judge=%.0fs probe=%.0fs probe_fired=%d",
        rollout_tag, len(records),
        summary["env_acc_per_traj"], summary["judge_success_rate"], summary["dropped_from_training"],
        summary["truncated_steps_total"], summary["parse_fail_steps_total"], term_counts,
        summary["wall_clock"]["total_s"]["mean"], summary["wall_clock"]["total_s"]["max"],
        summary["wall_clock"]["gen_total_s"]["mean"], summary["wall_clock"]["env_exec_total_s"]["mean"],
        summary["wall_clock"]["judge_total_s"]["mean"], summary["wall_clock"]["probe_total_s"]["mean"],
        summary["probe_fired_count"],
    )


def _agg_phase(records: list[dict[str, Any]], name: str) -> dict[str, float]:
    vals = [float((r.get("phases") or {}).get(name, 0.0) or 0.0) for r in records]
    return {
        "sum": round(sum(vals), 1),
        "mean": round(sum(vals) / len(vals), 1),
        "max": round(max(vals), 1),
    }
