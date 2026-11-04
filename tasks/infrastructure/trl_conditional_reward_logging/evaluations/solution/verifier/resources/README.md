# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`synchronized_steps` at step 24, with `score_mode=final`. Pass requires the
threshold 23.5 of 24. Historical defective baseline is 0 (the run crashes
before the first optimizer step).

Both ranks' metric streams are read. The custom `answer_parse_rate` metric
must be present at all 12 generation events on both ranks, on the same
steps, and equal the fraction of parsed answers in this run's
`rollouts_rank0.jsonl` rows at the preceding step, within absolute and
relative tolerance 0.000001 (the repaired trainer reports the mean of the
ranks that supplied values, and only rank 0 supplies one). Logging the
metric on rank 0 only, disabling it, replacing an absent rank contribution
with zero, or concentrating the rank-1 events on a subset of the 12 steps
fails the run.

The workspace copies of `run_training.sh` and `train.py` are byte-compared
against this verifier's pristine copies (`../run_training.sh`, `train.py`)
at grading time; a modified or missing runner fails the run. `train.py` is
byte-identical to the staged `run.py`, kept as a second copy so the
protected entry point is hash-anchored. Offline replays of packaged evidence
run without the stage environment and skip the check (reported as
`runner_integrity: "not checked (no stage environment)"`).

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace,
submits `/workspace/report.md`, and does **not** launch training. This
directory's `run_spec.json` drives the harness rerun stage — the command,
timeout, and log roots — executed in the agent's workspace before this
verifier runs. The harness deletes the declared log roots
(`experiment_logs`) before the rerun, so grading only ever sees this fresh
run's logs.

The runner fixes the model (`/workspace/model`), the data
(`/workspace/data`), seed 42, two GPU processes, 24 optimizer steps, four
completions per prompt, a 256-token completion limit, gradient accumulation
of two microbatches and four microbatches per generation batch, and offline
W&B, so the repair must live in `src/trl`.
