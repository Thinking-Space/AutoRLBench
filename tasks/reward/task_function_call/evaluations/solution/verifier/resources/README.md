# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It streams every log under `experiment_logs` line by line and assigns a weighted score:

- validation: the **mean** of `val-core/taco/acc/mean@1` over the pinned step
  set `200, 250, 300, 350, 400` must be at least `0.095` and at most `0.35` to
  score `1`, otherwise `0`. Every one of those five steps must be present; a
  run that stops early, resumes straight to the final step, or never evaluates
  fails this component with an explicit list of the missing steps.
- training reward: the mean `critic/rewards/mean` over training steps `1..400`
  must be strictly above `0.14` and at most `0.6` to score `1`, otherwise `0`.
  All 400 steps must be present.
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records the weighted float when only one component passes.

## Why a pinned-window mean and not a peak

The repaired run on the reference devspace produces this validation curve
(184 held-out problems, so one problem is `0.00543`):

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repaired | .0761 | .0924 | .1033 | .1196 | .1033 | .1141 | .1250 | .1087 | .0978 |
| defective | .0435 | .0652 | .0598 | .0598 | .0598 | .0870 | .0707 | .0870 | .0870 |

Three single-point statistics were rejected:

- **peak over all steps.** The repaired run's peak is `0.125` exactly once, so
  a bar at `0.125` is a coin flip on re-run; and because `TEST_FREQ` and
  `TOTAL_TRAINING_STEPS` are plain environment overrides in the launch script,
  a peak can be inflated by evaluating more often.
- **final step alone.** `0.0978` repaired against `0.0870` defective is a gap
  of two problems out of 184 — inside the run-to-run noise.
- **first step / step 0.** Measures the base model, not the repair.

The mean over the five pinned steps separates cleanly and averages the
sampling noise down by `sqrt(5)`: repaired `0.1098`, defective `0.0783`,
historical defective `0.0696`. The bar sits at `0.095`, roughly midway.

The ceilings (`0.35` validation, `0.6` training reward) exist because both
graded quantities are produced by `verl/utils/reward_score/taco.py`, the file
the agent is asked to repair. A scorer rewritten to return a constant `1.0`
would otherwise clear both floors.

## Log size

The repaired scorer executes candidate programs and ray forwards their stdout,
so a healthy 400-step run writes a console log two orders of magnitude larger
than a defective one (451 MB versus 4 MB on the reference run). The verifier
therefore never holds a log in memory: it iterates lines and skips any line
without verl's `step:` metric header. Grading the 451 MB log takes about 7 s at
20 MB peak RSS.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
