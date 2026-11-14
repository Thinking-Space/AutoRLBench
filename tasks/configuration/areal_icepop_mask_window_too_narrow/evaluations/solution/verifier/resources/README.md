`test.sh` is the single entry point. It does not use `submission.json`, an LLM, rules,
or rubrics. It finds the fresh run's `run/**/main.log` (regular files only, no symlinks,
newest first) and grades the `eval-rollout/reward` value the run logged at the end of its
third epoch — the primary held-out metric the frozen evidence also uses. This task's
experiment writes no generation dump the grader could re-score, so the logged value is
the scored value; it is trustworthy because the code that computes it is pinned.

- `metric_key`: descriptive only.
- `final_step`: `87` (3 epochs x 29 steps; `score_mode` is always final — the graded
  value is the last epoch's, never a mid-run peak).
- `min_eval_rows`: `3` — a complete run logs one `eval-rollout/reward` row per epoch-end
  evaluation; the graded row must follow the final `Train step 87/87 done.` line.
- `required_final_score`: `0.645` (no maximum; recalibrated 2026-11-13 from the original
  0.62 midpoint — see `oracle/README.md`). `historical_final_baseline`: `0.5777`,
  the defective run's final value.
- `runner_pinned`: `false`. The repair is an edit of `run.sh`, so the launch script is
  not byte-compared, as in `configuration/autorlbench_verl_grpo_rollout_top_p_off_policy`.
- `pinned_sha256` (503 files): the `areal/` source tree, `examples/` entry points,
  `pyproject.toml` + `uv.lock`, the GSM8K dataset files under `data/gsm8k/`, and the
  model directory's configuration / tokenizer files under
  `models/Qwen1.5-MoE-A2.7B-Chat/` (weight shards are not hashed — the model cannot be
  improved inside a trial, but its config, generation config and tokenizer must stay as
  staged).

The rerun is anchored so the graded value can only come from the shipped experiment:

- the files in `pinned_sha256` are checked against the workspace. This check runs when
  the stage sets `AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_VERIFIER` (live grading); a
  replay of packaged evidence by hand skips them.
- `main.log` is read as run-reality evidence too (never values beyond the graded one):
  the shipped runner's StatsLogger prints one `Train step N/87 done.` line per training
  step, so a complete run leaves exactly 87 of them, continuous 1..87, with the final
  `Train step 87/87 done.` line followed by the third evaluation row. A missing, short,
  gapped or eval-less log fails.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits
`/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json`
drives the harness rerun stage, `bash run.sh` with `RLROOT`, `WANDB_MODE`, `WANDB_DIR`
and the HF offline flags set, a 4 h budget (the recorded runs took about 1 h 40 m on
8 x A100), log root `run`, executed in the agent's workspace before this verifier runs.
The harness deletes the declared log root before the rerun, so grading only ever sees
this fresh run's logs.

`tests/grade_submission.py` is the grader of
`tasks/configuration/autorlbench_verl_grpo_rollout_top_p_off_policy` (v3 family) adapted
to AReaL's log-shaped evidence: `find_dumps` / `score_dump` / the parquet consistency
check are replaced by `find_main_logs` / `parse_run`, and the metric-key regex reads the
StatsLogger table cells (`│ key │ value │`) that `main.log` carries with ANSI colouring.
