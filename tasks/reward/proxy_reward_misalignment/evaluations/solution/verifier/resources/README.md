# Rule-based solution verifier

`test.sh` checks immutable assets and reward behavior, then extracts task metrics
only from fresh `$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B
files under `$AUTORLBENCH_RESULTS/experiment_logs/wandb`. It does not use an LLM
judgment or compare the submitted source to a required reference patch.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: the **terminal** `val-core/math__math/acc/mean@1` — the value at
  step 15, whose presence is required — must be at least `0.65` to score `1`,
  otherwise `0`. The terminal value is used, **not** a peak: the buggy run peaks
  at step 0 (`0.78125`) before the proxy corrupts the policy, so a peak selector
  would pass an unfixed rerun.
- training reward: the mean `critic/rewards/mean` over training steps must be
  strictly above `0.05` to score `1`, otherwise `0`. For this task the proxy
  reward is high by construction, so this component does **not** discriminate a
  fixed run from an unfixed one; the terminal validation accuracy is the real
  discriminator.
- final `score = 0.8 * validation + 0.2 * training reward`.

Pass requires both components (`score == 1.0`). The framework still records the
weighted float when only one component passes.

This grader is adapted from `infrastructure/torch_functional`. Two
deviations are required by the VIMPO PPO-Critic log format and this bug:

1. **Scalar reprs** — VIMPO prints validation scalars as numpy reprs
   (`val-core/math__math/acc/mean@1:np.float64(0.78125)`), and every validation
   dataset is on one `step:N - ...` line. `grade_submission.py` therefore
   tolerates the `np.float64(...)` wrapper and captures *every* val-core metric
   on each step line (the template captured only the first).
2. **Terminal, not peak** — `select_final` replaces the template's peak
   selector (see above).

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.

## Active v3 checks

Before applying the historical scalar thresholds, the verifier checks the data
and math-scorer hashes in `protected_assets.json` and 48 reward behavior cases.
Any failure scores zero. The v2 solution was run; v3 changes the training
questions and needs its own paired GPU qualification. The scalar thresholds
are retained for development comparison, not certified on v3. See the task
README and `SOLUTION_PROTOCOL_AUDIT.md` for remaining output-trust limitations.
