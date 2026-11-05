# Solution verifier

Run `test.sh` with `AUTORLBENCH_RESULTS` pointing to fresh run results.
The task-local grader reuses the repository's metric grader with one change:
step/metric matching is bounded to a single physical veRL console line, so
step 100 or 150 validation cannot attach to a preceding training step.

`val-core/openai/gsm8k/acc/mean@1` is accuracy on the prepared 64-question
held-out set. Passing requires a finite value >= 0.4 and <= 1.0 at exactly
step 150. It writes experiment_metrics.json, metrics.json and result.json.
No judge or native-gradient probe determines this full-training score.

The grader's `required_log_patterns` (in `resources/grader_spec.json`)
anchor the run's trigger configuration — per-token loss enabled, CP2, EP2
and `moe_z_loss_coeff` 0.001 must each appear in the fresh run's text logs
(they do, in the resolved-config dump). A repair that clears the metric bar
while disabling any of these fails, and all observed accuracy values must
stay within [0, 1].

Calibration uses the recorded run per arm plus a same-configuration rerun on
the shared image (2026-11-03, 2 GPUs): defective 0/0 in both runs; repaired
0.453125/0.546875 recorded and 0.515625/0.546875 rerun at steps 100/150.
Final margins to threshold are 0.4 and 0.146875; the smallest margin across
observed repaired points is 0.053125 (recorded step 100). The two runs share
seed and configuration but are not deterministic replays (step-1 grad_norm
223.2 vs 228.6), so the equal step-150 values are agreement between two
executions, not a variance bound. Cross-seed variance and step-0 validation
remain unavailable. A metric at step
0 or an earlier best step does not satisfy the final-step requirement.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. The task-root `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.
