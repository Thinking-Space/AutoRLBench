# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It runs `tests/grade_submission.py`, which reads two signals
from `$AUTORLBENCH_RESULTS/experiment_logs` and requires both.

## The two signals

The defect does not change what the policy learns per step — per matched step
the two arms are within noise. It changes **how much the engine gets done per
unit wall clock**: with the `abort -> in_place` override, interrupted rollout
groups decode to full length and are then dropped as stale, so decode compute is
wasted and fewer training steps finish in the fixed 8-hour budget. The primary
signal measures that loss directly; the secondary signal is its downstream
consequence on end-of-window quality.

- **primary — `train_steps`**: the largest `rollout N:` index + 1 across the
  driver logs. This is the direct, low-noise expression of the defect
  (historical 92 defective vs 148 control). Threshold **120**, calibrated at
  the midpoint: `+28` clear of the control, `−28` clear of the defective arm.
- **secondary — `rollout_solve_rate_last_window`**: mean `reward.acc` fraction
  over the last 10% of dumped rollout steps (`dump/rollout_data/*.pt`), read in
  the same raw-reward space the dumps were written in. Threshold **0.48**
  (defective 0.460, control 0.539): `+0.059` clear of the control, `+0.020`
  above the defective arm. It is thin and noisy (a last-10% window of ~9–15
  steps, each step's solve rate fluctuating ~0.4–0.7), which is why it is the
  **secondary** gate — the throughput signal does the real separation, and this
  corroborates that the extra steps turned into extra learning rather than a
  longer flat run.

Pass requires both. A run that is cut short (fewer steps) fails the primary
gate, and one whose solve rate never clears the bar fails the secondary gate, so
neither signal can be satisfied by a degenerate run.

## Calibration and its limits

One recorded run per side on 4×H800 for `TRAIN_SECONDS=28800`, same image and
assets, only the mixin override differing (`oracle/measured_runs/`). Because it
is one run per side, the margins above are reported without a variance estimate.

The primary threshold is **wall-clock absolute** and therefore hardware
dependent: the step count scales with the grading cluster's throughput, so the
invariant the defect actually fixes is the *ratio* (defective ≈ 0.62× the
control), not the absolute 92/148. `120` is correct for the 4×H800 setup the run
was calibrated on; on materially slower or faster hardware the threshold would
need to move with it. The run must not be shortened below the full 8-hour budget
(`instruction.md` fixes `TRAIN_SECONDS=28800`), because the solve-rate gap only
opens up late in training. These are the known limits of grading a
throughput-over-wall-clock defect; the issue level, which grades the attribution
directly, does not depend on them.

## Fixed recipe and wall clock (machine-checked)

The instruction fixes the dataset, model, rollout configuration and wall-clock
budget, and the grader enforces both from the fresh run's own logs, because a
repair that relaxes them instead of fixing the defect can restore the graded
numbers without doing the work:

- **launch recipe**: the ray entrypoint line must still carry
  `--pause-generation-mode abort`, `--max-weight-staleness 0`,
  `--rollout-batch-size 32`, `--n-samples-per-prompt 8`,
  `--rollout-max-response-len 8192` and `--global-batch-size 256`, and every
  resolved echo of those keys in the config dump must agree. Raising the
  staleness bound (interrupted groups then train instead of being dropped) or
  switching the pause mode is a throughput restoration, not a fix, and fails.
- **training window**: the span from the first to the last `rollout N:` step
  line's own timestamp must stay within `max_log_span_seconds` = 32400 s. The
  harness kills the command at 36000 s (task-root `run_spec.json`), and on the
  calibration hardware a defective run stretched to that cap reaches only
  ~115 steps — the 120-step threshold plus this cap keeps "run longer instead
  of fixing" closed, on grading clusters up to roughly 15% faster than the
  calibration cluster. Past that, recalibrate the absolute threshold with the
  cluster, as above.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`.
