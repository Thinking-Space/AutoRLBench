`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs
`val-aux/alfworld_valid_unseen/success/mean@1` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

The metric is the held-out unseen validation success rate, in `[0, 1]`.

- `metric_key`: `val-aux/alfworld_valid_unseen/success/mean@1`
- `score_mode`: `best` — the highest finite point over the run
- `required_final_score`: `0.6` (lower bound; the run must have trained with the
  schema aligned)
- `required_final_score_max`: `1.0` (upper bound; success is a rate, rejects
  fabricated values)
- `final_step`: `63` — the last eval of the fixed 63-step budget. Informational
  under `score_mode: best`.
- `historical_final_baseline`: `0.3333333333333333` — what the *defective*
  historical run logged at step 63. A repaired run must come in above it.

Pass is score `1`; all failures are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
