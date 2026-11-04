# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the shipped
(defective) checkout and its single-commit control.

Expected verdict: **unhealthy**.

## Root cause

- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py::DistBucketedWeightUpdateMixin._pause_and_prepare_engines`
  - kind: `symbol`
  - observed: when `self.args.pause_generation_mode == "abort"`, the method
    silently overrides it to `"in_place"` before pausing the rollout engines:

    ```python
    mode = self.args.pause_generation_mode
    if mode == "abort":
        mode = "in_place"
    ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
    if mode != "in_place":
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
    ```

    With `in_place`, in-flight requests are frozen and kept alive across the
    weight update instead of being aborted/evicted, and the subsequent
    `flush_cache` is also skipped (the `mode != "in_place"` guard). Those
    requests decode to completion against the old weights and are then discarded
    by the staleness filter (`--max-weight-staleness 0`), so the engine spends
    full-length decode compute on trajectories that never enter training.
  - expected: `mode` is left at the configured `pause_generation_mode`
    (`abort`), so the engines abort in-flight requests at weight sync and the
    cache is flushed. The launch config sets `--pause-generation-mode abort`;
    the code must honor it rather than rewrite it.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`outputs/driver.log` plus `outputs/rollout_data/*.pt`). The counterfactual
control-run figures are author-side reference and live in Documentation
reconciliation below.

- Primary source metric: train-step throughput over the fixed wall-clock
  budget. Only 92 `rollout N:` steps complete inside the fixed ~8-hour window
  (14:33 → 22:23), and most of each step is spent waiting on rollout decode:
  `perf/wait_time_ratio` averages ~0.82 across the run.
- Per-step `rollout/fully_async/stale_groups_recycled` sits at ~32 (a full
  rollout batch) every step while `rollout/fully_async/aborted_groups_recycled`
  stays at 0: interrupted groups are recycled as **stale** (decoded to full
  length, then dropped) rather than **aborted** (evicted mid-flight).
- The raw solve rate (fraction of rollouts with a correct final answer; readable
  from `rollout/raw_reward` in driver.log as (1 + raw_reward)/2, or computed by
  `extract_acc.py` on `rollout_data/*.pt`) rises from ~0.10–0.17 at step 0 to
  only ~0.40–0.45 by the end of the window — a downstream symptom of the
  throughput loss, not an independent failure.
- Frozen evidence is in `outputs/driver.log` and `outputs/rollout_data/*.pt`.

## Trigger conditions

- `pause_generation_mode` is `abort` (the launch default) and weight staleness
  is 0, so every weight sync interrupts the in-flight batch; the override turns
  each such interruption into wasted full-length decode instead of an eviction.

## Documentation reconciliation

- Control reference (author-side; same code, config, data and budget, only the
  `pause_generation_mode` override removed so the configured `abort` is
  honored). Rebuttal context only — the control run is not shipped in the
  workspace and comparisons to it are not evidence items: the control completes
  148 steps in the same window and reaches an end-of-window raw solve rate of
  ~0.54; per matched step the two arms learn comparably, the defective arm
  simply completes fewer steps per wall clock.
- The logged `pause_generation_mode ... abort` config line is **not** evidence
  of health: the defect is that the code ignores it and runs `in_place`. The
  symptom is visible only in the recycling counters and throughput, not in the
  config echo.
- Lower end-of-window solve rate and fewer completed steps are symptoms of the
  throughput loss, not independent root causes. "The run did not learn" / "the
  model collapsed" is a wrong attribution: both arms learn per step; the
  defective arm simply completes fewer steps per wall clock.
- `--kl-loss-coef 0.00`, `--entropy-coef 0.00`, `--rollout-max-response-len
  8192`, `--max-weight-staleness 0`, and the 4-GPU actor/rollout split are
  expected recipe settings, not defects.
