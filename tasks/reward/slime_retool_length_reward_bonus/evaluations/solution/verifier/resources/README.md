`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

`eval/aime` is read from the reward **score** channel. Without the length
bonus, correctness plus tool-call shaping has range `[-1.1, 1]`. The frozen
historical trace uses that channel plus the defective length bonus (`eval_reward_key: score`, see
`workspace/wandb/latest-run/files/config.yaml`).

- `metric_key`: `eval/aime`
- `score_mode`: `final` — the value at the run's last eval point
- `required_final_score`: `-0.55` (lower bound; the run must have trained)
- `required_final_score_max`: `0.5` (historical numerical gate; does not prove the length bonus is gone)
- `final_step`: `69` — the last eval of the fixed `--num-rollout 70` budget.
  The graded point under `score_mode: final`; a run that never reaches it
  scores `0`.
- `historical_final_baseline`: `3.0138113911946616` — what the *defective*
  historical run logged at step 69. It is above the upper bound, which is the
  point: a repaired run must come in **below** it, not above.

Pass is score `1`; all failures are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task-root `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
