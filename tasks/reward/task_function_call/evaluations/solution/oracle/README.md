# Canonical solution oracle

## Root cause

- `verl/utils/reward_score/taco.py::run_test`
  - kind: `symbol`
  - observed: `fn_name` is ignored and every sample is forced onto the stdin path; call-based compile / `Solution()` / `method(*inputs)` / return-value compare are deleted
  - expected: if `fn_name` is set, compile as call-based code and compare `method(*inputs)` against the expected return value; stdin scoring stays unchanged

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- dirty val curve 0.0217 → 0.0543 (peak 0.0870 at step 300)
- official function solutions score 0 on train and val; official stdin solutions still score 1
- val has 41/184 call-based problems, so those rows contribute 0 to `acc` even with a correct function solution

Repaired-versus-defective reference, measured on the same image and the same
8×H100 host (reference devspace `1236983`,
`/user/user2/harbor-runs/task_function_call__stage__env`):

| | step 200 | 250 | 300 | 350 | 400 | window mean | `critic/rewards/mean` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| repaired | .1033 | .1141 | .1250 | .1087 | .0978 | **0.1098** | **0.1838** |
| defective (re-run) | .0598 | .0870 | .0707 | .0870 | .0870 | 0.0783 | 0.1003 |
| defective (historical) | .0652 | .0652 | .0870 | .0761 | .0543 | 0.0696 | 0.0990 |

The "healthy scorer reaches 0.139 at step 400" figure quoted elsewhere in this
package comes from `tasks_coding_pass_01`, a different experiment on a
different stack (it had `pyext` installed; this task's `prepare.sh` installs it
deliberately not). This task's own repaired run never reaches it, so the
thresholds below are calibrated from the table above and not from that figure.

## Trigger conditions

- `verl/utils/reward_score/taco.py::run_test` is observed as ignoring `fn_name` and deleting call-based execution; the canonical expectation is restore call-based compile and `method(*inputs)` when `fn_name` is set.

## Required remediation

- `verl/utils/reward_score/taco.py::run_test`: `restore`; target state/value: if `fn_name` is set, use call-based compile and `method(*inputs)` return-value compare; stdin path unchanged.

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: the mean of `val-core/taco/acc/mean@1` over the pinned step set
  `200, 250, 300, 350, 400` must be within `[0.095, 0.35]`. All five steps must
  be present. The defective window mean is `0.0783`.
- training reward: mean `critic/rewards/mean` over training steps `1..400`
  must be strictly above `0.14` and at most `0.6`. All 400 steps must be
  present.
- final `score = 0.8 * validation + 0.2 * training reward`

The upper bounds are there because both graded quantities are emitted by the
very file the agent repairs; a scorer that returns a constant `1.0` clears
both floors. The window mean replaces the earlier peak statistic, which the
repaired run touched exactly once and which `TEST_FREQ` / `TOTAL_TRAINING_STEPS`
can inflate.

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.
