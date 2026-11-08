`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules, or rubrics,
and it does **not** read any logged metric: verl 0.9.0's Math-Verify validation scorer silently
under-reports on a busy host (process-pool queue time is charged against its 30 s per-sample timeout
and timed-out samples score 0), so `val-core/openai/gsm8k/acc/mean@1` is not load-independent. Instead
it finds the fresh run's held-out generation dump `experiment_logs/**/val_generations/100.jsonl`
(written by `trainer.validation_data_dir` as shipped, regular files only, no symlinks) and re-scores
every row with a fixed pure-Python rule: the number inside the last `\boxed{...}` of `output` (LaTeX
wrappers, `$`, `%`, commas removed, simple `\frac{a}{b}` evaluated) must equal `gts` numerically
(1e-6 relative tolerance). A response without a `\boxed{}` is wrong.

- `metric_key`: descriptive only (`heldout/gsm8k_boxed_acc@100 ...`). Nothing is parsed from logs
- `final_step`: `100`, the dump that is scored (`score_mode` is always final)
- `dump_subdir`: `val_generations`. `min_rows`: `1300` (the full 1,319-problem GSM8K test split)
- `required_final_score`: `0.53`. `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.3609`, the defective run's accuracy at step 100 under this rule

Pass is score `1`. All failures (missing / short / symlinked dump, accuracy below 0.53) are score `0`.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits
`/workspace/report.md`, and does **not** launch training. The task root's `run_spec.json` drives
the harness rerun stage, `mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_gsm8k_grpo.sh`
with `LOG_DIR`, `WANDB_MODE`, `WANDB_DIR`, `WANDB_SILENT` and a placeholder `WANDB_API_KEY` set, and
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` pinned to `1` so the rerun never
leaves the prepared image, a 4 h budget (the recorded runs took 2.4 to 2.7 h on
one RTX A6000), log root `experiment_logs`, executed in the agent's workspace before this verifier
runs. The harness deletes the declared log roots before the rerun, so grading only ever sees this
fresh run's dump.
