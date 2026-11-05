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

After you submit the report, the system clears `$AUTORLBENCH_RESULTS/experiment_logs`
and runs the following foreground command in the repaired workspace.

```bash
python3 launch.py
```

The run sets `LOG_DIR=$AUTORLBENCH_RESULTS/experiment_logs` and `WANDB_MODE=offline`.
It starts two independent local model services using the prepared checkpoints
`models/Qwen3-4B` and `models/Qwen3-8B`, and executes the telecom configuration in
`configs/rollouts.json`. Both services share one 80-GiB GPU with separate 0.42 memory
fractions. The agent uses 8192 completion tokens and the user simulator uses 512.
The horizon is four task IDs crossed with seeds 1 through 4, with `max_steps=200`
and four scheduled trajectories. Requests to each model are serialized.

The system allows 3600 seconds including service startup. The verifier reads only this fresh run, requires the declared 16-trajectory horizon and at least 12 error-free trajectories,
and checks API compatibility. Submit your report after repairing the workspace.
