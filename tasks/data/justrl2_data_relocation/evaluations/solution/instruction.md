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

After you submit `report.md`, the harness clears
`$AUTORLBENCH_RESULTS/experiment_logs` and runs:

```bash
mkdir -p "$AUTORLBENCH_RESULTS/experiment_logs" && bash run.sh
```

The environment contains `LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs` and
`WANDB_MODE=offline`. The prepared public model and data are under
`./prepared/`; the run writes `experiment_logs/run-home`.

Keep the original model, all 64 training and 16 fixed evaluation questions,
question/answer bytes, PPO recipe, eight updates, evaluation interval four,
8 groups × 2 samples, and 32768-token response budget. The scheduled storage
migration after the first four updates must remain. Do not change the reward
function, synthesize outcomes, delete source IDs, disable the migration, or
suppress required events. Cheap local data-loading checks are allowed.

The grader checks the fresh run's data integrity and completion, not a noisy
held-out accuracy threshold. It requires both difficulty strata in each
post-migration training batch, successful backing-file access, all eight real
actor and critic updates, unchanged original inputs, final optimizer
checkpoints and all fixed evaluation snapshots. It independently recomputes
these checks from the recorded events and files. Hiding failed reads or
changing reported counters does not pass.
