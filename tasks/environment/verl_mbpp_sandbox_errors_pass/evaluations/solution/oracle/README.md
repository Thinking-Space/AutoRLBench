# Canonical solution oracle

## Root cause

- `reward/mbpp_reward.py` (`_SERVER`, the exec loop of the training sandbox)
  - kind: `symbol`
  - observed: only `AssertionError` marks a sample wrong; other exceptions pass, so unfenced or
    broken responses earn full reward and the policy drops its code block
  - expected: every exception counts as a failure

## Observable impact

- Primary source evaluation metric: `val-core/google-research-datasets/mbpp/acc/mean@1`, 0.2786 at
  step 0 and 0.0 from step 10; training score 1.0 from step 11.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `reward/mbpp_reward.py`: `except BaseException: ok = False` in the server loop (`fix.diff`). Any
  sandbox that rewards only programs that exit cleanly after all asserts is equivalent.

Machine grading reads only the new experiment logs. The grader re-executes
`experiment_logs/val_generations/60.jsonl` (last ```python block, setup and all asserts, fresh
interpreter, 10 s timeout, at least 490 rows) and passes at ≥ 0.33; it also requires
`metrics.jsonl` to cover steps 0-60 and the raw MBPP files to match their pinned sha256.
