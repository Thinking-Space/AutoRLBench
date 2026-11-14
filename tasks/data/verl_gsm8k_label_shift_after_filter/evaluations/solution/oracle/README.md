# Canonical solution oracle

## Root cause

- `data_prep/gsm8k.py::main` (train-split filter)
  - kind: `symbol`
  - observed: the 500-character filter is applied to `questions` only; `zip(questions, answers)`
    pairs 96.5% of training questions with another problem's answer
  - expected: questions and answers filtered together

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`; held-out accuracy re-scored
  from `val_generations` stays at 0.4488-0.4655; training score 0.017 on average.

## Trigger conditions

- Same as the issue oracle.

## Required remediation

- `data_prep/gsm8k.py`: keep the index list and filter both lists with it (`fix.diff`, the control's
  code). Any construction that pairs every training question with its own answer is equivalent.

Machine grading reads only the new experiment logs. The grader re-scores
`experiment_logs/val_generations/60.jsonl` (last number of each response vs the gold answer, at
least 1,300 rows) and passes at ≥ 0.51; it also requires `metrics.jsonl` to cover steps 0-60
and every row of the regenerated training parquet to be a raw GSM8K training question with its own
gold answer (at least 7,000 rows).
