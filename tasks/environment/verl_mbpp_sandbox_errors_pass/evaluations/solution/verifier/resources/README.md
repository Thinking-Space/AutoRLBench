`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules, or rubrics,
and it does not score any logged metric. It finds the fresh run's held-out generation dump
`experiment_logs/**/val_generations/60.jsonl` (written by `trainer.validation_data_dir` as shipped,
regular files only, no symlinks) and re-scores every row with a fixed harness (`"rule": "mbpp_exec"`):
the last ```python block of `output` is run with the problem's setup code and all of its asserts
(from `gts`) in a fresh Python interpreter with a 10 s timeout, and the row passes iff the process
exits 0. The workspace's reward code is not used.

- `metric_key`: descriptive only. Nothing is parsed from logs
- `final_step`: `60`, the dump that is scored (`score_mode` is always final)
- `dump_subdir`: `val_generations`. `min_rows`: `490` (the 500-problem MBPP test split; one prompt is
  dropped by `data.filter_overlong_prompts`)
- `required_final_score`: `0.33`. `required_final_score_max`: `1.0`
- `historical_final_baseline`: `0.0000`, the grader's score of the defective run at step 60
- `runner_pinned`: `true`; the runner is byte-compared against the pristine copy in `resources/`
  (the defect lives in `reward/mbpp_reward.py`, not the runner, so the runner must not change), and
  the 60-step budget is additionally held by the `metrics.jsonl` check
- `pinned_sha256`: the four raw MBPP parquets under `raw/mbpp/full/`
- `train_parquet_consistency` / `test_parquet_consistency` (both `kind: mbpp`): every row of the
  regenerated `data/mbpp/{train,test}.parquet` must be one raw problem — `reward_model.ground_truth`
  is the JSON of exactly the raw problem's `test_setup_code` and `test_list` (matched by `task_id`),
  and the prompt is exactly one user turn built from the raw `text` and first test with the pinned
  template. The training split's `raw` lists the three source parquets. The runner rebuilds both
  parquets from raw on every rerun, so the scored validation set is anchored the same way as the
  training set: a swapped or weakened test suite, or a prompt carrying anything beyond the template
  (e.g. the solution), fails grading

The pinned files, the runner and both regenerated parquets (tests and prompt form) are checked
against the workspace when the stage sets `AUTORLBENCH_WORKSPACE` /
`AUTORLBENCH_VERIFIER` (live grading; a replay of packaged evidence by hand skips them), and
`metrics.jsonl` is read as run-reality evidence only (one JSON line per training step `0..60` and a
`val-core/google-research-datasets/mbpp/acc/mean@1` row at step 60).

## System rerun (`run_spec.json`)

The harness runs `mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run_grpo_mbpp.sh` from
the workspace root with `OUT=$AUTORLBENCH_RESULTS/experiment_logs` and `WANDB_MODE=offline`, a 3 h
budget (the recorded runs took about 30 minutes on one A100), log root `experiment_logs`. The
harness deletes the declared log roots before the rerun.

`tests/grade_submission.py` is the grader of `tasks/configuration/verl_grpo_adamw_eps_too_large`
with optional spec keys added: `rule` (`mbpp_exec` here), `runner`, and `val_key` (the logged
validation key that marks the final evaluation in `metrics.jsonl`).
