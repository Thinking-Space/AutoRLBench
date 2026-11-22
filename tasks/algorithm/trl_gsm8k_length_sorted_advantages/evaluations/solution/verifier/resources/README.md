# Scoring resources

`../test.sh` is the single entry point. It does not use an LLM, rules or
rubrics, and it does not read any logged metric value.

- `grader_spec.json`: `final_step` 100; the scored dump is
  `experiment_logs/eval_after.json.items.jsonl` (written by the shipped
  `eval_gsm8k.py`, one row per GSM8K test item in file order, 1,319 rows
  required); `required_final_score` 0.60; `historical_final_baseline` 0.0000,
  the grader's score of the shipped defective run.
- `gsm8k_test_gold.json`: the gold answers of the GSM8K main test split in
  file order, taken from the parquet whose sha256 is pinned in
  `grader_spec.json`. A row is correct only if `finish == "stop"` and the
  number after its last `####` (commas, `$` and a trailing period removed)
  equals its gold answer numerically.
- `pristine/`: byte-exact copies of the shipped `run_grpo.sh`,
  `train_grpo.py`, `eval_gsm8k.py`, `common.py` and `check_answer.py`.

Run-reality checks, never scored: `metrics.jsonl` must contain a training row
(with `loss`) for every global step 1-100 and the trainer's summary row
(`train_runtime`) at step 100, and `eval_after.log` must contain the
`eval 100:` line.

Live grading (the stage sets `AUTORLBENCH_WORKSPACE`) also byte-compares the
five runner files against `pristine/` and checks the two GSM8K parquets by
sha256, so the repair has to be in `workspace/trl`. A replay of packaged
evidence by hand runs without the stage environment and skips these checks.

Offline replay: `AUTORLBENCH_RESULTS=<dir> bash ../test.sh`, where
`<dir>/experiment_logs` holds a run's logs. For the recorded control,
copy `../../oracle/recorded_control/experiment_logs` into `<dir>`; it passes
with 0.7384. The six recorded buggy 100-step runs score 0.0000-0.4268 and
fail; the three controls score 0.7210-0.7468 and pass.
