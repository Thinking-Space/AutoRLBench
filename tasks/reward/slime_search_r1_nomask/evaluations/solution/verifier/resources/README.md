`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$AUTORLBENCH_RESULTS/experiment_logs`.

- `metric_key`: `eval/nq_test`, exact-match accuracy in `[0, 1]`
- `score_mode`: `best` — the highest finite point over the run
- `required_final_score`: `0.28` (lower bound; same bar as the healthy twin)
- `required_final_score_max`: `1.0` (accuracy cannot exceed 1; blocks a
  fabricated value)
- `final_step`: `99` — the last eval of the instruction's `--num-rollout 100`
  budget. Informational under `score_mode: best`.
- `historical_final_baseline`: `0.15433333333333332` — the frozen defective
  run's step-99 value.

Pass is score `1`; all failures are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
