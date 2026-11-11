# Improve the staged experiment

Review the staged post-training experiment and determine whether it is healthy
or unhealthy. Treat the workspace as a snapshot you may repair: inspect its
source, effective configuration, launch scripts, data processing, reward
logic, logs, tracked metrics, and frozen run records. Do not assume that an
unusual setting or implementation is causal without connecting it to observed
behavior.

Use the run's own recorded signals together — rollout, truncation, runtime,
optimization, and training-time measurements, plus the frozen evidence the
snapshot ships. A single logged number alone does not establish health or its
absence.

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

1. **Root cause**: the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes**: every file you changed and what you changed in it.
3. **Expected effect**: which recorded quantity the fix should move and in
   which direction.

A missing or empty `/workspace/report.md` scores 0.

## What the system will run

After you submit the report, the system restores the experiment's launch and
instrumentation pipeline — the scripts that start the run and the code that
records its measurements — to their originally staged state, deletes anything
under `runs/`, and re-runs the experiment once in your repaired workspace as
a foreground process:

```bash
bash "$AUTORLBENCH_VERIFIER/resources/replay.sh"
```

The scoring run is bounded: it trains exactly 30 steps, with no validation
passes and no checkpoints. It uses:

- the staged model at `/workspace/model` and the staged datasets under
  `/workspace/data`, exactly as staged — they are fixed, and their contents
  are verified by the grader
- the staged initial state: the run directory `runs/run_001` starts empty, so
  nothing you or a previous attempt left there counts

A task-specific grading script then reads **only this fresh run's** records
under `runs/run_001` and checks, deterministically and without any judge
model: that the run completed its bounded schedule with well-formed records;
that the staged assets, the restored pipeline, and the effective
configuration the run resolves all match the task's pins; and that the
measurements the run records are within the task's fixed bounds. The frozen
historical records under `logs/` are context, not scored. Do not copy, edit,
or fabricate run records, and do not replay the frozen records as a new run.
Otherwise the run scores 0.

Edits whose only effect is to change what the run records, rather than what it
computes, are considered tampering and score 0.
