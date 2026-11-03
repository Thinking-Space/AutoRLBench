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

- Primary source metric: train-step throughput over the fixed wall-clock budget
  (number of `rollout N:` steps completed). The defective run completes far
  fewer steps in the same wall clock than the control.
- Per-step `rollout/fully_async/stale_groups_recycled` sits at ~32 (a full
  rollout batch) every step while `rollout/fully_async/aborted_groups_recycled`
  stays at 0: interrupted groups are recycled as **stale** (decoded to full
  length, then dropped) rather than **aborted** (evicted mid-flight).
- Because fewer steps are completed per unit time, the raw solve rate (fraction
  of samples with `reward.acc`) reaches a lower value by the end of the fixed
  wall-clock window, even though per matched step the two arms are comparable.
- Frozen evidence is in `outputs/driver.log` and `outputs/rollout_data/*.pt`.

## Trigger conditions

- `pause_generation_mode` is `abort` (the launch default) and weight staleness
  is 0, so every weight sync interrupts the in-flight batch; the override turns
  each such interruption into wasted full-length decode instead of an eviction.

## Documentation reconciliation

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
