# Canonical solution oracle

## Root cause

- `reward/gsm8k_reward.py::compute_score` (`_SCORE_CACHE`)
  - kind: `symbol`
  - observed: memo keyed by `(data_source, split, index)`; every rollout of a prompt gets the first
    rollout's score, so GRPO advantages are zero and the policy never moves
  - expected: each response scored on its own

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` (frozen at 0.4337 by the memo);
  held-out accuracy re-scored from `val_generations` stays at 0.4655-0.4655.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `reward/gsm8k_reward.py`: remove the memo (`fix.diff`, which restores the control's plain call of
  `gsm8k.compute_score(..., method="flexible")`), or key it by the response text as well.

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer, at
least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to cover steps 0-60
and the regenerated training parquet to match the pinned raw GSM8K train split.
