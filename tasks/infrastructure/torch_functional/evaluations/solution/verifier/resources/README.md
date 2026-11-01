# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: the highest finite `val-core/taco/acc/mean@1` at any step must be at least `0.125` (12.5%) to score `1`, otherwise `0`. A later lower step does not fail this component if an earlier peak already meets the threshold.
- training reward: the mean `critic/rewards/mean` over training steps must be strictly above `0.14` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records the weighted float when only one component passes.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
