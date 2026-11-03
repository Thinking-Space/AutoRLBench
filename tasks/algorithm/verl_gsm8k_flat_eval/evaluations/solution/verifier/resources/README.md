# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a binary score: the finite `val-core/openai/gsm8k/reward/mean@1` at step 50 must be at least `0.74609375` to pass (`score == 1.0`), otherwise `0`.

The threshold was calibrated from one observed run per arm: the defective arm's maximum was `0.734375`, the baseline arm's minimum was `0.75390625`, and the baseline workspace's step-0 score was `0.73828125`. The threshold is `0.01171875` above the defective maximum, `0.0078125` below the baseline minimum, and `0.0078125` above step 0. These margins describe the recorded runs only; one run per arm does not establish variance or statistical significance.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`. It uses no LLM judge at the solution level.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
