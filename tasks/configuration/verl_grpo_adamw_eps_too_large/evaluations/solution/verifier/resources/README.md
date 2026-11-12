`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules, or rubrics,
and it does not score any logged metric. It finds the fresh run's held-out generation dump
`experiment_logs/**/val_generations/100.jsonl` (written by `trainer.validation_data_dir` as shipped,
regular files only, no symlinks) and re-scores every row with a fixed rule: the last
`#### <number>` in `output`, after removing commas / `$` / a trailing period, must equal `gts`
numerically (1e-6 relative tolerance).

- `metric_key`: descriptive only. Nothing is parsed from logs
- `final_step`: `100`, the dump that is scored (`score_mode` is always final)
- `dump_subdir`: `val_generations`. `min_rows`: `1300` (the full 1,319-problem GSM8K test split)
- `required_final_score`: `0.53`. `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.493`, the grader's score of the defective run at step 100
- `runner_pinned`: `false`. The repair is an edit of `run_gsm8k_grpo.sh`, so the runner is not
  byte-compared, as in `configuration/verl_grpo_ppo_mini_batch_too_small`
- `pinned_sha256`: both `data/gsm8k/*.parquet` and the two raw GSM8K parquets

The rerun is anchored so the scored dump can only come from the shipped experiment:

- the files in `pinned_sha256` are checked against the workspace. This check runs when the stage sets
  `AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_VERIFIER` (live grading); a replay of packaged evidence by
  hand skips them.
- `metrics.jsonl` is read as run-reality evidence only (never its values): the shipped runner sets
  `VERL_FILE_LOGGER_PATH`, so a real run leaves one JSON line per training step `0..100` and a
  `val-core/openai/gsm8k/acc/mean@1` row at step 100. A missing, short or gapped file fails.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits
`/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives
the harness rerun stage, `mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_gsm8k_grpo.sh`
with `LOG_DIR`, `WANDB_MODE` and `WANDB_DIR` set, a 3 h budget (the recorded runs took 40 to 60 min
on one RTX PRO 6000), log root `experiment_logs`, executed in the agent's workspace before this
verifier runs. The harness deletes the declared log roots before the rerun, so grading only ever
sees this fresh run's dump.

`tests/grade_submission.py` is the grader originally built for
`tasks/reward/verl_grpo_gsm8k_any_number_reward` (same runner and dump format; that task
was removed from the benchmark on 12 Nov 2026), with optional spec keys added, of which
this task uses one: `runner_pinned: false` skips the runner byte-compare.
