# Diagnose and repair the staged training setup

The workspace holds an asynchronous agentic RL training setup: the framework source, the launcher
and its configuration, the archived console logs and metrics of earlier runs of the same workload on
a larger cluster, and the logs of one run of this workspace.

Determine whether the setup is healthy. If it is not, find the root cause in the code and
configuration, and repair it in the staged workspace at `/workspace`. Do not assume a setting is
causal without connecting it to observed behaviour.

**Do not launch training.** Do not start the experiment in the foreground or the background, and do
not use `nohup`, `&`, `tmux`, `screen` or any detach mechanism. The system runs the experiment
itself after you finish; anything you launch is wasted and scores nothing. Cheap non-training checks
(reading code and logs, `--print-command`, unit-level snippets) are fine.

**Do not use the network.** Diagnose and repair with what is staged locally.

When you are done, submit `/workspace/report.md` with the verdict and the evidence that decides it,
the root cause and the evidence connecting it to the observed behaviour, every file you changed and
what you changed in it, and which metric your fix should move and in which direction. A missing or
empty report scores 0.

## What the system will run

The system clears its own log directory and runs the workspace's launcher as
`PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/train_reproduction.py`: the bundled Miles
fully-asynchronous trainer on 8 GPUs (4 training, 4 serving), the pinned Qwen3.5-35B-A3B policy, and
three rollouts of the staged workload.

The run's shape is fixed and graded from the fresh run's own logs. These workspace files are
byte-compared against pristine copies and must not be modified: everything under `scripts/` and
`runtime/` other than `scripts/run_config.json`. In `scripts/run_config.json` only the `router`
section may change; the model, topology, training, rollout and environment sections are compared
against the pristine configuration. Everything else in the workspace, including the vendored
framework source, may be repaired.

Grading requires the run to complete three rollouts with a successful optimizer update each, to keep
the workload intact (session count and multi-turn depth), and then measures what fraction of
follow-up turns are served by an engine that still holds the previous turn's prompt in its prefix
cache, computed from the per-turn records the run writes. Only files from this fresh run are graded;
do not copy, edit or fabricate metrics.
