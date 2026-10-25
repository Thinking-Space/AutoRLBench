# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, retrieval, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system runs the experiment itself after you finish (see below);
anything you launch is wasted and scores nothing — and it leaves a GPU busy or
a stale run-name behind, which can make the graded rerun refuse to start.
Cheap, non-training checks are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
python3 run.py --run-name graded --runs-dir "$AUTORLBENCH_RESULTS/experiment_logs"
```

from `/workspace`. That entry point owns the whole run: it starts the local
retrieval server on the port in `config.json`, waits for it to answer, streams
training output, and shuts the server down when training ends or is
interrupted. Do not bypass `run.py` — the graded run always goes through it.

The effective run uses:

- base model `/workspace/shared/models/Qwen2.5-1.5B-Instruct`
- train data `/workspace/shared/data/train.parquet`
- validation data `/workspace/shared/data/test.parquet`
- retrieval corpus and index `/workspace/shared/data/{corpus.jsonl,e5_Flat.index}`
  with the query encoder `/workspace/shared/models/e5-base-v2`
- the budget already in `config.json`: `steps` 32, `eval_every` 10,
  `batch_size` 32, `learning_rate` 1e-06 — do not change these
- `WANDB_MODE=offline` (`run.py` sets this; nothing logs in to W&B)

`python3 run.py --check` validates the interpreter, the assets, the local
imports and the data row counts without a GPU or any training; it is the cheap
way to confirm the workspace is runnable before you submit.

Two guards in `run.py` will refuse to start rather than produce a bad run: it
needs at least 10 GiB of free disk, and it refuses a GPU that already has
memory in use. This is one more reason not to launch anything on the GPU
yourself, and not to create a `graded` run directory ahead of time — the graded
rerun uses `--run-name graded` and refuses a run-name that already exists.

`run.py` creates `experiment_logs/graded/` with `train.log`,
`retrieval_server.log`, `retrieval.jsonl`, `rollouts/`, `run_config.json`,
`exit_status.json` and the offline W&B directory. A task-specific grading
script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs written
anywhere else are ignored. Do not copy, edit, or fabricate log metrics.
Otherwise the run is scored 0.
