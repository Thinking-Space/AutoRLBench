`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules, or rubrics,
and it does **not** read any logged metric: the task's defect is the reward function that produces
`val-core/openai/gsm8k/acc/mean@1`, so that number cannot be trusted. Instead it finds the fresh
run's held-out generation dump `experiment_logs/**/val_generations/100.jsonl` (written by
`trainer.validation_data_dir` as shipped, regular files only, no symlinks) and re-scores every row
with a fixed rule: the last `#### <number>` in `output`, after removing commas / `$` / a trailing
period, must equal `gts` numerically (1e-6 relative tolerance).

- `metric_key`: descriptive only (`heldout/gsm8k_strict_acc@100 ...`). Nothing is parsed from logs
- `final_step`: `100`, the dump that is scored (`score_mode` is always final)
- `dump_subdir`: `val_generations`. `min_rows`: `1300` (the full 1,319-problem GSM8K test split)
- `required_final_score`: `0.28`. `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.0`, the defective run's strict accuracy at step 100

Pass is score `1`. All failures (missing / short / symlinked dump, accuracy below 0.28) are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits
`/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives
the harness rerun stage, `mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_gsm8k_grpo.sh`
with `LOG_DIR`, `WANDB_MODE` and `WANDB_DIR` set, a 4 h budget (the recorded runs took 2.1 to 2.7 h on
one RTX A6000), log root `experiment_logs`, executed in the agent's workspace before this verifier
runs. The harness deletes the declared log roots before the rerun, so grading only ever sees this
fresh run's dump.
