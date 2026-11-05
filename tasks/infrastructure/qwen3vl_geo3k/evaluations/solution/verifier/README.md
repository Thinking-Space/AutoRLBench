# Solution verification

Entry: `test.sh`; run through the AutoRLBench verifier stage with the pinned image.
Only the new foreground experiment under `$AUTORLBENCH_RESULTS/experiment_logs`
is accepted as training evidence. The recorded historical logs are not a new run.

The score is 1 only if the 20-step completion/gradient/validation/W&B checks
pass and fresh six-case GPU parity passes. Every engine position must match,
and every case's maximum absolute token log-prob difference must be below
1.0 nat. This metric is independent of the Geo3K reward score.

One run per side supplied calibration. The frozen defective maximum was
3.7402865886688232 nat; the control maximum was 0.550462007522583 nat.
Margins to the 1.0 limit are 2.7402865886688232 and 0.449537992477417 nat.
Validation accuracy and reward have no independent pass threshold.
The separate fresh-run completion requirement prevents a source-only repair
from passing. Parity is measured with the pinned original checkpoint and the
submitted implementation, rather than the trained step-20 checkpoint.

Training evidence is checked for steps 1–20, finite gradient norms with at
least one positive value, validation at two or more distinct steps, and an
offline W&B file. These checks inspect logs; they do not authenticate their
origin or enforce every declared hyperparameter.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task-root `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
