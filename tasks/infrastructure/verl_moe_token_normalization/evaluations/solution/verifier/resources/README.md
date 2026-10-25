# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`val-core/openai/gsm8k/acc/mean@1` at step 150, with `score_mode=final`.
Pass requires accuracy >= 0.4 and <= 1.0. Historical defective baseline is 0.
No auxiliary metric is weighted. The accuracy key is a fraction, not percent.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
