#!/usr/bin/env python3
"""Dump every held-out episode's reward, keyed by task, as the eval runs.

slime's own eval logging reports means: `eval/et_test` is the pass rate over the
whole split, and that is all it keeps. A mean is not enough to compare two
training runs here. The two arms are scored on the *same* tasks, so the right
test is a paired one -- per task, did this arm solve it more often than the
other? -- and a paired test needs the per-task numbers, which the mean has
already thrown away. Across a 300-task split an unpaired comparison of two
~6% pass rates needs several times the episodes to reach the same power,
because almost all of the variance between the arms' means is task difficulty
that is identical on both sides and cancels when you subtract per task.

So this hook writes one JSONL row per episode next to the run's other
artifacts. It returns a falsy value on purpose: slime treats a truthy return as
"the custom logger handled it", and the aggregate lines are still wanted.

Wired in by run.sh as
`--custom-eval-rollout-log-function-path et.eval_log.log_eval_rollout_data`.
"""

from __future__ import annotations

import json
import os

# Written by the RolloutManager actor, not by the driver, so the path has to
# travel through the Ray runtime env rather than being inherited from run.sh.
DUMP_DIR = os.environ.get("ET_EVAL_DUMP_DIR", "")


def _status(sample) -> str:
    status = getattr(sample, "status", None)
    return getattr(status, "value", str(status))


def _reward(sample, fallback):
    """The scalar reward, whatever shape the sample carries it in.

    slime allows `Sample.reward` to be a dict (`--reward-key` picks a field out
    of it). `data[name]["rewards"]` has already been resolved to scalars, so it
    is the authoritative source; the sample is only consulted if the two lists
    somehow fail to line up.
    """
    if fallback is not None:
        return fallback
    reward = getattr(sample, "reward", None)
    return reward if not isinstance(reward, dict) else None


def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None):  # noqa: ARG001
    """slime hook: `custom_eval_rollout_log_function_path`.

    Returns None so slime goes on to emit its own aggregate `eval N:` line --
    this adds a record, it does not replace one.
    """
    if not DUMP_DIR:
        return None
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"eval_rollout_{rollout_id:04d}.jsonl")

    with open(path, "w", encoding="utf-8") as fh:
        for name, payload in data.items():
            samples = payload.get("samples") or []
            rewards = payload.get("rewards") or []
            for i, sample in enumerate(samples):
                row = {
                    "rollout_id": rollout_id,
                    "dataset": name,
                    # The task directory name, which is what makes the two arms'
                    # rows joinable and the comparison paired.
                    "label": getattr(sample, "label", None),
                    "index": getattr(sample, "index", None),
                    "group_index": getattr(sample, "group_index", None),
                    "reward": _reward(sample, rewards[i] if i < len(rewards) else None),
                    "response_length": getattr(sample, "response_length", None),
                    "status": _status(sample),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return None
