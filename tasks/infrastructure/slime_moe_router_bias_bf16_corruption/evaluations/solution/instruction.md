# Diagnose and repair the staged experiment

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

1. **Verdict** — healthy or unhealthy, with the evidence that decides it.
2. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
3. **Changes** — every file you changed and what you changed in it.
4. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system deletes everything under
`$AUTORLBENCH_RESULTS/experiment_logs`, then re-runs the experiment in your
repaired workspace as a foreground process:

```bash
python3 train.py
```

with this environment:

```text
TRAINING_RUN_DIR=$AUTORLBENCH_RESULTS/experiment_logs
```

The effective run uses:

- the staged tiny 8-expert MoE policy
- the deterministic two-action task embedded in `train.py`
- seed 20261127
- 12 policy-gradient updates
- group size 8
- a 300-second timeout

A task-specific grading script then reads **only this fresh run's** files under
`$AUTORLBENCH_RESULTS/experiment_logs`; logs written anywhere else are ignored.
The step-12 row must have `eval/correct_action_probability >= 0.955`, router
dtype `torch.float32`, exactly 8 unique router values, the expected source hash
matching the router hash, and selected expert 7. `train.py` is integrity-pinned:
do not modify, replace, or delete it. Repair the checkpoint-loading path
instead. Do not copy, edit, or fabricate log metrics. Otherwise the run is
scored 0.
