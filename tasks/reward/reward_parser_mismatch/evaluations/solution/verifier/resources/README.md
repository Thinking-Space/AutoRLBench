# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a validation-first binary
score (`score_mode: final` in `grader_spec.json`):

- validation: the **terminal** `val-core/math__math/acc/mean@1` — the value at the latest step `<= final_step` — must be at least `0.10` (10%). The terminal value is used, **not** a peak, so a run that briefly rises and then collapses does not pass. (For this bug the metric is `0.0` at every step, so terminal and peak coincide at `0.0`.) Whenever the fresh run logged any validation point, this gate alone decides the outcome.
- training reward: the mean `critic/rewards/mean` over training steps strictly above `0.05` is only a fallback gate for runs that produced no validation metric at all.
- final `score = 1.0` on pass and `0.0` otherwise.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

This grader is adapted from `infrastructure/torch_functional` with two deviations required by the VIMPO PPO-Critic log format: it tolerates numpy scalar reprs (`...:np.float64(0.0)`) and captures every val-core metric on each `step:N` line (the template captured only the first), and it selects the terminal value via `select_final` rather than the peak.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
