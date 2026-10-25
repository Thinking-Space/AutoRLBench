# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system runs the experiment itself after you finish (see below);
anything you launch is wasted and scores nothing. Cheap, non-training checks
are fine.

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
bash scripts/train.sh
```

with this environment:

```text
TRAINING_RUN_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses:

- model `/workspace/models/Qwen1.5-MoE-A2.7B-Chat`
- train data `/workspace/data/gsm8k/train.parquet` (7473 official training rows)
- validation data `/workspace/data/gsm8k/validation64.parquet` (the prepared 64 held-out rows)
- 150 training steps, validation at step 100 and the final step 150
- the supplied configuration, seeds, CP2/EP2/TP1 layout and two allocated GPUs
- `WANDB_MODE=offline`

The launcher places stdout under `experiment_logs/logs/train.log` and offline
W&B data under `experiment_logs/wandb`. The final held-out validation must be
logged at step 150; a step-0 score or an earlier best score is not sufficient.
A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs` and scores the key metrics; logs written
anywhere else are ignored. Keep the dataset, validation selection, reward
function, model, batch sizes, training budget and evaluation schedule
unchanged. Do not disable configured loss terms or parallelism to avoid an
implementation defect. Do not copy, edit, or fabricate metrics or replay frozen
historical logs as a new run. Otherwise the run is scored 0. Leave frozen
evidence under `observed_run/` intact.
