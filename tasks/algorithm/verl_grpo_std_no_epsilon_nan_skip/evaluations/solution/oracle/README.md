# Canonical solution oracle

## Root cause

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
  - kind: `symbol`
  - observed: `(score - mean) / std` without epsilon → NaN advantages for zero-variance groups →
    NaN gradient → every optimizer step skipped
  - expected: `(score - mean) / (std + epsilon)`

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`; held-out accuracy re-scored
  from `val_generations` stays at 0.4617-0.4723; `actor/grad_norm` NaN at every step.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `verl/trainer/ppo/core_algos.py`: restore `+ epsilon` (`fix.diff`). Any variant that gives
  zero-variance groups a finite advantage (e.g. `algorithm.norm_adv_by_std_in_grpo=False`) is
  equivalent for grading.

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer, at
least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to cover steps 0-60
and the regenerated training parquet to match the pinned raw GSM8K train split.
