# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and recorded rollout trajectories. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Base the diagnosis on training reward anomalies and rollout trajectories,
including action formatting, tool interactions, termination, and truncation.
Connect these observations to the source code and effective configuration.
A reward drop or truncated log alone is not proof of a particular root cause.

## Your job

If you find a defect, fix it in the staged workspace at `/workspace`. **Do not
launch training.** Do not start the experiment, in the foreground or
background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism. The system runs the experiment itself after you finish (see below);
anything you launch is wasted and scores nothing. Cheap, non-training checks
are fine. The dataset, held-out set, model, training-step budget, and
evaluation interval are fixed.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After the report is submitted, the system clears `$AUTORLBENCH_RESULTS/experiment_logs` and runs:

```text
python run.py
```

This invokes the historical `scripts/train_4_gpus.sh` with the console log,
Hydra output, and offline W&B data rooted below
`$AUTORLBENCH_RESULTS/experiment_logs`, checkpoints and rollout records
rooted below `$AUTORLBENCH_RESULTS/runtime_outputs`, and disables checkpoint
resume. Offline W&B data is created below
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`. The run is scored only from
files under `$AUTORLBENCH_RESULTS/experiment_logs`; logs written anywhere
else are ignored. The effective run must use:

- model `/workspace/models/Qwen3-VL-4B-Thinking`
- the six staged tool-service models under `/workspace/models/`
- train data `/workspace/data/route/train_with_zwz_modified_prompt.parquet`
- validation data `/workspace/data/route/test_subset_100_seed42.parquet`
- `trainer.total_training_steps=200`
- `trainer.test_freq=40`
- `LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs`
- `WANDB_MODE=offline`
- `WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs`

The verifier byte-compares the pinned runner files (`run.py`, the `scripts/`
launch stack, `tools/log_paths.py`, and
`verl/verl/utils/logger/aggregate_logger.py`) against its pristine copies at
grading time; modifying them fails the run. Do not change the reward function,
the datasets, the model, or the training budget. Repair the staged rollout
construction code as the diagnosis indicates.

This case scores recovery from training collapse using the mean training score
at step 200. Only files from this fresh run are graded. Do not copy, edit, or
fabricate log metrics. Otherwise the run is scored 0.
