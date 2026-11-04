# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`correct_objective_steps` at step 24, with `score_mode=final`. Pass requires
the threshold 23.5 of 24. Historical defective baseline is 12, whose passing
steps compare zero-valued objectives, so the margin rests on the nonzero
comparisons.

The score recomputes the accumulation-window DAPO objective from this fresh
run's own token records (`token_records/*.npz`): per-token ratios, advantages
and completion masks must reproduce each record's logged loss within relative
tolerance 0.0005 and absolute tolerance 0.000002, and both microbatches of
every optimizer step must pass.

Three provenance checks close record-only shortcuts: the run-shape metadata in
each record (world_size=1, accumulation_steps=2, generation_steps=4) is
pinned to the configuration the runner fixes; the four microbatches of each
generation batch must have mask token counts summing to the recorded
generation_token_count; and each step's logged trainer loss must equal the
sum of that step's two recorded microbatch losses, so the record loss fields
cannot be rewritten without also forging the metric stream.

## Runner integrity

`run_spec.json`'s command is `bash run_training.sh`, executed in the agent's
workspace: the harness re-runs the experiment with the workspace's own
runner, which imports the repaired TRL tree from `src/`. The workspace copies
of `run_training.sh` and `train.py` are byte-compared against this
verifier's pristine copies (`../run_training.sh`, `train.py`) at grading
time; a modified or missing runner fails the run. `train.py` is
byte-identical to the staged `run.py`, kept as a second copy so the protected
entry point is hash-anchored. Offline replays of packaged evidence run
without the stage environment and skip the check (reported as
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
(`/workspace/data`), seed 42, one GPU process, 24 optimizer steps, four
completions per prompt, a 256-token completion limit, gradient accumulation
of two microbatches and four microbatches per generation batch, and offline
W&B, so the repair must live in `src/trl`.
