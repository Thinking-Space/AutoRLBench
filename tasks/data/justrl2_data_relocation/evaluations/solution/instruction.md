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

The incident is already materialized before you begin. Inspect `incident/`, the
recorded logs, and `loader.py`. Submit your change to `loader.py` and `report.md`.
Its interface is `resolve(root, record) -> relative_path`; the harness reads and
checks the returned file independently. The requested logical sample must remain
the same. Transient reads may be retried; substituting a different sample does not
repair the run.

The harness restores both roles from the preserved checkpoint and continues the
remaining four updates. It uses the immutable engine and snapshot under
`$AUTORLBENCH_PREPARED`, consumes only your submitted `loader.py`, and writes fresh
artifacts to `$AUTORLBENCH_RESULTS/experiment_logs/run-home`. It does not execute
modified workspace launch scripts, metrics code, reward code or model code.

Passing requires the planned logical samples and original content to reach
training, actual checkpoint/model/Adam/scheduler/RNG restoration, four subsequent
updates and valid final checkpoints. A reward increase is not required. Keep
prepared assets unchanged and do not run training yourself.
