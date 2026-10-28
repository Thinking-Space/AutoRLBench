# Solution verifier

Run `test.sh` with `AUTORLBENCH_RESULTS` pointing to fresh run results.
The task-local grader is the `verl_moe_token_normalization` grader with one
addition: the step-60 validation dump is re-graded with the verifier's own copy
of verl's GSM8K scorer (`tests/gsm8k_reference.py`) against a protected answer
key (`resources/gsm8k_test_answer_key.json`). All 1319 test problems must appear
exactly once, answers come from the key rather than the dump's `gts`, and the
logged metric must agree with the re-grade within 0.002. A rerun whose reward
function, scorer, dump or logged numbers were edited in the workspace therefore
cannot pass on its own data. Whether the generations themselves come from this
rerun is left to the harness's execution constraints.

`val-core/openai/gsm8k/acc/mean@1` is greedy accuracy on the 1319-question
GSM8K test split with verl's strict `####` answer extraction. Passing requires a
finite value >= 0.4 and <= 1.0 at exactly step 60 (`score_mode=final`: a
step-0 score or an earlier best does not count). It writes
experiment_metrics.json, metrics.json and result.json.

Calibration uses 2 runs per arm (seed 1 on 1 GPU, seed 2 on 2 GPUs; 60 steps
each). Step-60 values: defective 0.08415 / 0.08112, repaired 0.7771 / 0.7794.
Across all 13 validation points of both defective runs the maximum is 0.09098;
across both repaired runs from step 5 on the minimum is 0.7263. Both repaired
runs start at 0.08112 / 0.07961 at step 0, so the threshold also clears the
repaired arm's step 0 by 0.319 (the fix alone, without training, does not pass).
Margins at step 60: 0.40 - 0.08415 = 0.316 above the defective arm, and
0.7771 - 0.40 = 0.377 below the repaired arm. This is an initial threshold
calibrated on two seeds per side, not a claim of cross-environment stability.

Offline replay of this grader on the recorded runs: both defective runs exit 1
(below 0.4), both repaired runs exit 0, and on all four every row matches the
key (1319 of 1319) and the re-graded accuracy equals the logged one. Tampered
result directories all exit 1: a single fabricated row with a logged 1.0, a
single real problem answered correctly with a logged 1.0, one problem replaced
by a duplicate of another, an extra unknown row, `gts` replaced by the model's
own answers with the log edited to match, and a logged metric edited to 0.9.
