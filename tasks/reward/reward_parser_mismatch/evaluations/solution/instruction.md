# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward logic,
logs, tracked metrics, and frozen evaluation evidence. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health — a reward that rises while held-out accuracy falls is a warning,
not a success.

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
repaired workspace as a foreground process, using the launch script:

```text
run_buggy.sh
```

with this environment:

```text
MODEL_PATH=/workspace/Qwen3-4B-Base
CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs
LOGGER=console
WANDB_MODE=offline
WANDB_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses the fixed training budget baked into the launch script
and its config; do not change it. Offline W&B data will land below
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`. A task-specific grading script then
reads **only this fresh run's** files under `$AUTORLBENCH_RESULTS/experiment_logs`
and scores the key metrics; logs written anywhere else are ignored. Make sure
your repaired workspace still writes its training and evaluation logs into
`$AUTORLBENCH_RESULTS/experiment_logs`. Do not copy, edit, or fabricate log
metrics. You may repair `autorlbench_case/buggy_reward_fn.py`. Keep the frozen
assets and historical logs under `autorlbench_case/` unchanged, and do not modify
held-out labels or the evaluation protocol. Otherwise the run is scored 0.

Completion is required independently of metric quality: validation at the
declared final step must be present, and evaluation beyond the fixed horizon
is not accepted. Where training reward is graded, every training step must be
logged. Accuracy and binary rewards must remain in their valid [0, 1] range.
