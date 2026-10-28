# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`val-core/openai/gsm8k/acc/mean@1` at step 60, with `score_mode=final`.
Pass requires accuracy >= 0.4 and <= 1.0. Historical defective baseline is
0.08415. No auxiliary metric is weighted. The accuracy key is a fraction, not
percent.

The step-60 validation generations (`experiment_logs/val_generations/60.jsonl`)
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

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$AUTORLBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.

The command appends Hydra overrides after the launch script's own, so the
model, data files, batch size, sampling width, LoRA rank/alpha/merge, FSDP2
strategy, reward, validation decoding, step budget and evaluation schedule are
the recorded ones even if the agent edited `run_grpo_lora.sh`.
