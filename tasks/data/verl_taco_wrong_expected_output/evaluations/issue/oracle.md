# Canonical issue oracle

Expected verdict: **unhealthy**.

> CALIBRATION STATUS: the task was rebuilt on Qwen3-4B-Instruct-2507
> (2026-11-09) and both arms were recorded 2026-11-10 (defective arm through
> step 49 of 50 -- the PREEMPTABLE host was reaped during step-50/final
> validation with a healthy log tail, so the recorded evidence is complete
> through the 40-step grading window; control arm at the full 40 steps). The
> numeric sections below are filled from those runs.

## Root cause

The training labels are natively wrong for a large fraction of the training
problems, so correct solutions are deterministically scored 0. Both parts are
essential:

1. **Label noise, not difficulty.** `workspace/taco_verl/train.parquet` mixes
   ~40% TACO train problems from the unverified dirty pool (problems for which
   no bundled solution passes the problem's own tests) with verified-clean
   problems at matched `EASY`/`MEDIUM` difficulty. In the dirty rows the
   expected output of at least one test is wrong in the dataset itself: the
   problems' own bundled solutions -- which agree with each other -- fail that
   test. The validation set (`test.parquet`) is verified-clean, so the training
   reward is suppressed while held-out evaluation stays meaningful.
2. **Causal chain to the failure.** The TACO reward is binary (1 only when
   every test passes). On a dirty problem, a rollout that implements the
   correct algorithm still fails the natively-wrong test and earns 0, so GRPO
   assigns correct rollouts a negative (or zero) advantage: the model is
   trained to discard correct algorithms for those problems. The training
   reward mean stays pinned below the clean baseline scaled by the clean
   fraction, and the rate of externally-correct-but-scored-0 rollouts stays
   nonzero instead of vanishing as training progresses.

The difficulty mix being `EASY`/`MEDIUM` with a 4B instruct policy that
solves ~34% of the matched held-out split at step 0 is how the agent can tell
this is not the sibling failure "training data too hard": the same model,
same scorer, on matched-difficulty verified problems learns, while a
persistent subset of training groups produces agreeing correct-looking
output that the tests reject.

## Observable impact (from the recorded Qwen3-4B-Instruct-2507 runs; the
## defective arm ran 49/50 steps, the control arm the shipped 40 steps)

- `critic/rewards/mean`: defective arm 0.370 averaged over the run (range
  0.09-0.65, no upward trend; last five recorded dumps 0.431) versus the
  clean control's 0.602 (range 0.38-0.86, last five 0.627). The defective
  mean sits at the control's mean scaled by the 0.6 clean share
  (0.6 x 0.602 = 0.361), i.e. pinned there instead of tracking the clean
  problems' own learnability.
- Per-problem training reward on the dirty subset stuck at 0 while the
  external judge (agreement with the problem's own agreeing bundled
  solutions) rates a nonzero fraction correct: in the final five dumps of
  the 40-step window the externally-correct-but-scored-0 rows per step were
  47/25/36/41/18 out of 128. The two scorers separating on the dirty
  subset -- and only on it -- is the defect's fingerprint; the grader's
  headline number is the max over the last five dumps of
  (externally-correct-but-scored-0 rows / 128): 0.3672 for the defective
  arm versus 0.0000 for the control arm (dumps 36-40 on both sides).
- Correct-but-zero rollouts receiving negative (or zero) advantage: see
  `evaluations/solution/oracle/control_reference/group_advantage.json`
  (share of dirty GRPO groups in the final five dumps containing an
  externally-correct rollout with reward 0, split by negative vs zero
  advantage).
- Held-out accuracy on the clean 184-row `EASY`/`MEDIUM` validation set:
  defective arm stagnates at its step-0 level (0.353 -> 0.332 -> 0.293 ->
  0.359 -> 0.332 -> 0.364 -> 0.353 -> 0.359 -> 0.337 -> 0.348 over
  validations 0-45, mean 0.343, no trend) while the control arm improves
  from the same 0.32 step-0 level to 0.359 at step 40 (0.321 -> 0.321 ->
  0.380 -> 0.332 -> 0.364 -> 0.337 -> 0.337 -> 0.353 -> 0.359).

## Symptoms, not root causes

- Reward sparsity / many all-zero GRPO groups: partially present (the dirty
  problems contribute all-zero groups even when solved correctly), but the
  clean majority of groups behaves normally.
- Any per-test partial-credit behavior, format problems, or parse failures.

## Not the root cause

- Data difficulty mismatch (the sibling task `verl_taco_difficulty_mismatch`):
  the training rows here are `EASY`/`MEDIUM`, matched between dirty and clean
  rows; the model demonstrably solves the clean ones.
- Reward or code defects: `verl/utils/reward_score/taco.py`, the launch
  script, the model, and the training stack are the healthy baseline. The
  binary all-tests-pass reward is the standard TACO reward; the defect is in
  what it is fed, not in how it scores.
- KL / entropy settings, learning rate, batch shape.
- Model capacity: held-out accuracy and clean-problem reward both move
  (and the model solves a third of the held-out set at step 0).
- Reward hacking: hard-coded outputs may appear as a coping strategy, but the
  persistent correct-but-zero rate on the dirty subset is the primary channel.

## Evidence

- `workspace/taco_verl/train.parquet` rows carry no provenance label; the
  dirty/clean distinction is not a column. It is established by execution:
  run (or reason about) candidate solutions against each row's own tests, or
  cross-check each row's tests against the worked examples in its problem
  statement. A diagnosis that only asserts "some labels are wrong" without
  connecting it to the reward/advantage chain earns partial credit.
