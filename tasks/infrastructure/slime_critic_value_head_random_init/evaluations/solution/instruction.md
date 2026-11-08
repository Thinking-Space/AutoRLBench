# RL training run diagnosis and repair

Review the staged post-training experiment in `/workspace` and determine
whether it is healthy or unhealthy. Treat the workspace as a historical
snapshot: inspect its source, effective configuration, launch scripts, data
processing, reward logic, logs, tracked metrics, and frozen evaluation
evidence. Do not assume that an unusual setting or implementation is causal
without connecting it to observed behavior.

Use held-out evaluation evidence together with rollout, truncation, runtime,
optimization, and training signals. Training reward alone does not establish
held-out health. A truncated log is not by itself proof of a defect.

If you conclude that the run is unhealthy, repair the cause in the
workspace's own code. The dataset, held-out set, model, training-step
budget, evaluation interval, reward function, and rollout configuration
are fixed — repair the implementation, not the task.

## Your job

**Do not launch training — do not run `./run.sh` yourself, in the foreground
or background, and do not use `nohup`, `&`, `tmux`, `screen`, or any detach
mechanism.** The system runs the experiment itself after you finish (see
below); a run you start is wasted and scores nothing.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Verdict** — healthy or unhealthy, with the evidence that decides it.
2. **Root cause** — the concrete cause of any problem, with the causal
   chain from it to the observed behavior, distinguishing root causes from
   symptoms.
3. **Changes** — every file you changed and what you changed in it.
4. **Expected effect** — the metric your change should move and its
   expected value.

A missing or empty `/workspace/report.md` scores 0.

## How you will be checked

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then runs the experiment end-to-end in
your (possibly repaired) workspace (a full run needs 2 GPUs and takes
roughly 15 minutes; the Endless Terminals prompt set ships prebuilt with the
workspace, so there is no first-run build pass) and a task-specific verifier
grades the fresh results, including structural checks that the run actually
trained, generated rollouts, and produced non-empty responses.

The verifier reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored.
Do not copy, edit, or fabricate log metrics, and keep `run.sh` writing its
artifacts where it does now (`OUT_DIR`, which the system points at
`$AUTORLBENCH_RESULTS/experiment_logs`). Otherwise the run is scored 0.
Cheap, non-training checks are fine.

**Do not use the network.** Do not fetch code, documentation, packages,
or any other material from outside the staged workspace — diagnose and
repair using only what is staged locally.
