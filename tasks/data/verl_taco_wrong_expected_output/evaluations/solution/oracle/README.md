# Canonical solution oracle

## Root cause

- `taco_verl/train.parquet` (assembled by the prepare stage)
  - kind: `data`
  - observed: ~40% of the rows are TACO train problems from the unverified
    dirty pool whose expected outputs are natively wrong -- the problems' own
    bundled solutions agree with each other on an output and still fail their
    own tests. Under the binary all-tests-pass reward, a correct rollout on
    such a row earns 0 and GRPO pushes the policy away from correct algorithms
    for those problems.
  - expected: training rows whose tests a correct solution can actually pass
    (the verified subset), difficulty-matched to the shipped mix.

## Required remediation

Remove the natively-wrong-expected-output rows from `taco_verl/train.parquet`
(or replace them with verified-clean rows at matched `EASY`/`MEDIUM`
difficulty). Concretely, the repaired `train.parquet` must:

- keep the taco_verl schema, prompt template, and `data_source` unchanged;
- keep at least `min_train_rows` rows (1,200, from the grader's
  `grader_spec.json`): the dirty rows are ~40% of the shipped mix, so dropping
  them alone leaves ~828 rows, below the floor -- replace them with
  verified-clean rows (or repeat clean rows), or repair the dirty rows' labels
  in place;
- stay disjoint from `test.parquet` prompts (checked by the grader);
- leave `test.parquet`, the launch script, and `verl/utils/reward_score/taco.py`
  byte-identical (checked by the grader);
- keep a non-empty `/workspace/report.md` (checked by the grader).

Identifying the dirty rows without network access is part of the task: a row's
own tests can be executed against candidate solutions offline (the workspace
model can generate candidates, and the worked examples inside each problem
statement are already a consistency check against its test cases).

Machine grading reads only the new experiment logs. The grader re-scores the
fresh run's `experiment_logs/rollout_data/` dumps: it executes each dumped
response on the test of its own problem and, on the natively-wrong-expected
rows, against the corrected output (what the problem's own agreeing solutions
produce). Pass requires the "externally-correct but scored 0" rate over the
last five step dumps to be at or below `required_final_score_max` in
`grader_spec.json`. The defective arm's rate is materially above zero and does
not fall with training; a clean run's is ~0. Logged metric values and
symlinked dumps are ignored. The grader also anchors the repaired
`train.parquet` against blinding: every row must keep the shipped prompt
template, at most `max_residual_dirty_rows` rows may still carry one of the
recorded wrong-expected-output signatures (a row whose problem statement was
rewritten to hide it from the lookup while its wrong label stayed in place
counts exactly like an unrepaired one), and score-0 dump rows that no longer
parse back to a problem fail the run.

## Recorded comparison (`control_reference/`)

Both arms recorded 2026-11-10 on Qwen3-4B-Instruct-2507, 4xH800, verl GRPO
(response budget 16384, 16 prompts x 8 rollouts, validation every 5):

- **Defective arm** (as shipped, 49/50 steps recorded before a platform
  reaper killed the PREEMPTABLE host during final validation; the 40-step
  grading window is fully covered): train reward mean 0.370 (last five
  0.431), held-out accuracy flat at the step-0 level (mean 0.343 over
  validations 0-45, no trend), correct-but-scored-0 rate over dumps 36-40
  per step 0.367/0.195/0.281/0.320/0.141, headline (max) 0.3672.
- **Control arm** (train.parquet rebuilt from verified-clean rows only,
  EASY 1100 / MEDIUM 280, same seed 20261106, sha256
  `887ef2f2…`, 40 steps): train reward mean 0.602 (last five 0.627),
  held-out accuracy 0.321 -> 0.359, correct-but-scored-0 rate 0.0000 over
  dumps 36-40.

The grader threshold `required_final_score_max` = 0.02 sits between the two:
margin to the control's rate 0.020 (lower bound), margin to the defective
run's 0.3672-0.02 = 0.347 (upper bound). One run per side; margins are
reported without variance. Full numbers in
`control_reference/historical_metrics.json` and
`control_reference/group_advantage.json`.
