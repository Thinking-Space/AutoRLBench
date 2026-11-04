# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `verl/utils/reward_score/taco.py::run_test`
  - kind: `symbol`
  - observed: `fn_name` is ignored and every sample is forced onto the stdin path; call-based compile / `Solution()` / `method(*inputs)` / return-value compare are deleted
  - expected: if `fn_name` is set, compile as call-based code and compare `method(*inputs)` against the expected return value; stdin scoring stays unchanged

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- Historical defective validation rises from 0.0217 to 0.0543 (peak 0.0870 at step 300).
- The same-package repaired rerun ends at 0.0978; its mean over steps
  200/250/300/350/400 is 0.1098, versus 0.0783 for the paired defective rerun.
  The separate `tasks_coding_pass_01` result of 0.139 used a different stack
  with `pyext` and is not a matched control for this package.
- official function solutions score 0 on train and val; official stdin solutions still score 1
- val has 41/184 call-based problems, so those rows contribute 0 to `acc` even with a correct function solution

## Trigger conditions

- `run_test` no longer reads `in_outs.get('fn_name')` and always sets `CODE_TYPE.standard_input`.
- The call-based compile / `method(*inputs)` branch is absent, so a correct `def add(a, b): return a + b` is wrapped as stdin `code()` and fails.

## Documentation reconciliation

- Parquet and prompts still contain call-based problems (`fn_name`, `Use Call-Based format`); the defect is scorer-only.
- `prime_code` and agent-loop tool parsers are unused by this TACO launch and are distractors.
