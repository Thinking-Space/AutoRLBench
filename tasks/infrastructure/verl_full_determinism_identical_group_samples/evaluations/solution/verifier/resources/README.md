# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`val-core/openai/gsm8k/acc/mean@1` at step 30, with `score_mode=final`.
Pass requires accuracy >= 0.4 and <= 1.0. Historical defective baseline is
0.08719. No auxiliary metric is weighted. The accuracy key is a fraction, not
percent.

The step-30 validation generations (`experiment_logs/val_generations/30.jsonl`)
are re-graded with `../tests/gsm8k_reference.py`, a verbatim copy of verl's
`verl/utils/reward_score/gsm8k.py` (strict `####` extraction), against the
protected answer key `gsm8k_test_answer_key.json`: one entry per GSM8K test
problem (1319, no duplicate questions), keyed by the sha256 of the user message
with whitespace collapsed, built from the prepared `data/gsm8k/test.parquet` by
`build_answer_key.py`. Each dump row is matched to the key by its question;
every key problem must appear exactly once, and rows that match no key problem,
missing problems or repeated problems fail the run. Answers come from the key,
not from the dump's `gts`. The re-graded accuracy must be within 0.002 of the
logged metric and must also be >= 0.4.

`grader_spec.json` adds two checks on the training rollout dumps
(`rollout_dump`, `experiment_logs/rollouts/<step>.jsonl`):

- `rollout_n`, `rollout_groups`, `max_identical_group_share`: every step from 1
  to 30 must have 64 prompt groups of 8 rows, and fewer than 0.5 of the groups
  may have all 8 `output` strings identical.
- `repeat_root`, `repeat_steps`: `experiment_logs/repeat/rollouts/<step>.jsonl`
  for steps 1, 2 and 3 must hold exactly the same rows (compared as a multiset
  of JSON objects) as the main run's. Nothing under `repeat/` is read for the
  metric.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.

The command runs `run_grpo_fd.sh` twice with the same Hydra overrides appended
after the launch script's own: 30 steps into `experiment_logs`, then 3 steps
into `experiment_logs/repeat`. The model, data files, data seed, batch size,
sampling width and temperature, GRPO estimator, full-determinism flags, rollout
seed, attention backend, eager mode, single replica, reward, validation
decoding, step budget and evaluation schedule are the recorded ones even if the
agent edited `run_grpo_fd.sh`.
