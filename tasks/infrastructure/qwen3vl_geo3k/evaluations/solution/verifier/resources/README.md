# Grader inputs

Run the parent `test.sh`, not an agent-supplied metrics file. Fresh experiment
logs must be below `$AUTORLBENCH_RESULTS/experiment_logs`.

`grader_spec.json` defines the metric
`per_response_token_logprob_abs_diff_nat`, six cases, 64 response tokens,
1.0 nat repair limit, and 20 required training updates. Correct positions
are mandatory. The severe-candidate 0.25 nat and paired 2x limits belong to
the paired reference check. They are not solution-score requirements or
reward-improvement targets.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
