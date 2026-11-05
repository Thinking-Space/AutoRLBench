# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: the mean finite `val-core/taco/acc/mean@1` over the pinned steps `200, 250, 300` must land in `[0.095, 0.35]` to score `1`, otherwise `0`. All three steps must be present, which is what the designated 300-step run at `TEST_FREQ=50` produces; a truncated run scores `0`.
- training reward: the mean `critic/rewards/mean` over training steps `1..300` must land in `(0.155, 0.6]` to score `1`, otherwise `0`. All 300 steps must be present.
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records the weighted float when only one component passes. The ceilings guard against a scorer rewritten to return a constant reward, since both graded quantities come out of the file the agent is asked to repair.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task root's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
