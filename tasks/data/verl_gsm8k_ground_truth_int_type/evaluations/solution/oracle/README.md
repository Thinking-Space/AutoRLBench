# Canonical solution oracle

## Root cause

- `data_prep/gsm8k.py::build_rows`
  - kind: `symbol`
  - observed: `ground_truth` stored as `int(float(...))`; verl's GSM8K scorer compares the extracted
    answer string with `==`, so every reward is 0
  - expected: the string label

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` (0.0 at every evaluation);
  held-out accuracy re-scored from `val_generations` stays at 0.4602-0.4701.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `data_prep/gsm8k.py`: store `extract_solution(answer)` as a string (`fix.diff`). Making the reward
  compare numerically is equivalent for grading.

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer,
numerically, at least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to
cover steps 0-60 and the regenerated training parquet to match the pinned raw GSM8K train split.
