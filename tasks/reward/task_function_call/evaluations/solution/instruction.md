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

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process, using the launch script:

```text
examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh
```

with this environment:

```text
MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct
TRAIN_FILE=/workspace/taco_verl/train.parquet
TEST_FILE=/workspace/taco_verl/test.parquet
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
WANDB_API_KEY=offline
TOTAL_TRAINING_STEPS=400
TEST_FREQ=50
```

Offline W&B data will land below
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`. A task-specific grading script then
reads **only this fresh run's** files under `$AUTORLBENCH_RESULTS/experiment_logs`
and scores the key metrics; logs written anywhere else are ignored. Make sure
your repaired workspace still runs correctly under these exact settings — for
example, if you change where logs are written, keep them going to
`$AUTORLBENCH_RESULTS/experiment_logs`. Do not copy, edit, or fabricate log
metrics. Otherwise the run is scored 0.
