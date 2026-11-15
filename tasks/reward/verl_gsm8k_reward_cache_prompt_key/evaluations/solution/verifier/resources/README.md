`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules, or rubrics,
and it does not score any logged metric. It finds the fresh run's held-out generation dump
`experiment_logs/**/val_generations/60.jsonl` (written by `trainer.validation_data_dir` as shipped,
regular files only, no symlinks) and re-scores every row with a fixed rule (`"rule": "last_number"`):
the last number in `output`, after removing commas, must equal `gts` numerically (1e-6 relative
tolerance). This is the quantity the task's reward measures (verl's `flexible` GSM8K extraction),
computed without the workspace's reward code.

- `metric_key`: descriptive only. Nothing is parsed from logs
- `final_step`: `60`, the dump that is scored (`score_mode` is always final)
- `dump_subdir`: `val_generations`. `min_rows`: `1300` (the full 1,319-problem GSM8K test split)
- `required_final_score`: `0.51`. `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.4617`, the grader's score of the defective run at step 60
- `runner_pinned`: `false`; the 60-step budget is held by the `metrics.jsonl` check
- `pinned_sha256`: the two raw GSM8K parquets under `raw/gsm8k/main/`
- `train_parquet_consistency`: every row of the regenerated `data/gsm8k/train.parquet` must be a
  raw training question carrying its own gold answer, with at least 7,000 rows

The rerun is anchored so the scored dump can only come from the shipped experiment: the pinned files
and the training parquet are checked against the workspace when the stage sets
`AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_VERIFIER` (live grading; a replay of packaged evidence by hand
skips them), and `metrics.jsonl` is read as run-reality evidence only (one JSON line per training
step `0..60` and a `val-core/openai/gsm8k/acc/mean@1` row at step 60).

## System rerun (`run_spec.json`)

The harness runs `mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_grpo_gsm8k.sh` from
the workspace root with `OUT=$AUTORLBENCH_RESULTS/experiment_logs` and `WANDB_MODE=offline`, a 3 h
budget (the recorded runs took 26-37 minutes on one A100 40 GB or 80 GB), log root `experiment_logs`. The
harness deletes the declared log roots before the rerun.

`tests/grade_submission.py` is the grader of `tasks/configuration/verl_grpo_adamw_eps_too_large`
with two optional spec keys added: `rule` (`last_number` here; `strict` keeps the original
`#### <number>` rule) and `runner` (the runner file name for the optional byte-compare).
