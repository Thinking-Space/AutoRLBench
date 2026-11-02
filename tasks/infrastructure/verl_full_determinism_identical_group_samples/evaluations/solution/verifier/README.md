# Solution verifier

Run `test.sh` with `AUTORLBENCH_RESULTS` pointing to fresh run results.
The task-local grader is the `verl_moe_token_normalization` grader with three
additions, all reading only `experiment_logs`:

1. The step-30 validation dump is re-graded with the verifier's own copy of
   verl's GSM8K scorer (`tests/gsm8k_reference.py`) against a protected answer
   key (`resources/gsm8k_test_answer_key.json`). All 1319 test problems must
   appear exactly once, answers come from the key rather than the dump's
   `gts`, and the logged metric must agree with the re-grade within 0.002.
2. Group diversity: for each training step 1 to 30, `rollouts/<step>.jsonl`
   must hold 64 prompts with 8 responses each, and fewer than half of the
   groups may consist of 8 identical responses. The check does not require 8
   distinct responses per group; short or easy answers may legitimately
   coincide.
3. Reproducibility: the rerun repeats the same command for 3 steps under
   `experiment_logs/repeat/`, and `repeat/rollouts/<step>.jsonl` must contain
   exactly the same rows as `rollouts/<step>.jsonl` for steps 1, 2 and 3.
   Files under `repeat/` are not read for the metric.

`val-core/openai/gsm8k/acc/mean@1` is greedy accuracy on the 1319-question
GSM8K test split with verl's strict `####` answer extraction. Passing requires
a finite value >= 0.4 and <= 1.0 at exactly step 30 (`score_mode=final`: a
step-0 score or an earlier best does not count). It writes
experiment_metrics.json, metrics.json and result.json; metrics.json also lists
the identical-group count per step and the repeat comparison.

Calibration uses 2 runs per arm (data seeds 1 and 2, one GPU, 30 steps each).
Step-30 values: defective 0.08719 / 0.08719, repaired 0.7384 / 0.7354. All 14
defective validation points are 0.08719; across both repaired runs from step 5
on the minimum is 0.6543. Both repaired runs start at 0.08719 at step 0, so the
threshold also clears the fix without training. Margins at step 30:
0.40 - 0.08719 = 0.313 above the defective arm and 0.7354 - 0.40 = 0.335 below
the repaired arm. Identical groups: 64 of 64 at all 30 steps in both defective
runs, 0 of 64 at all 30 steps in both repaired runs. This is an initial
threshold calibrated on two seeds per side, not a claim of cross-environment
stability.

Offline replay of this grader on the recorded runs (each 30-step run paired
with a recorded 3-step run of the same arm as its repeat): both repaired runs
exit 0, and both defective runs exit 1 on the accuracy, re-grade and diversity
checks while passing the reproducibility check. A repaired 30-step run paired
with the 3-step run of an earlier, request-id-seeded version of the fix (diverse
groups, different streams on every run) exits 1 on the reproducibility check
alone. Tampered result directories also exit 1: a single fabricated validation
row with a logged 1.0, a logged metric edited to 0.9, `gts` replaced by the
model's own answers, a missing repeat, a repeat dump with one row changed, and a
rollout dump in which most groups were made identical.
