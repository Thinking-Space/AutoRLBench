# Diagnose and repair the staged training setup

The workspace holds an asynchronous agentic RL training setup: the framework source, the launcher
and its configuration, the sandbox runtime and the local sandbox service the rollout uses, the
archived console logs and metrics of earlier runs of the same workload on a larger cluster, and the
logs of one run of this workspace.

Determine whether the setup is healthy. If it is not, find the root cause or causes in the code and
configuration, and repair them in the staged workspace at `/workspace`. Do not assume a setting is
causal without connecting it to observed behaviour, and do not stop at the first contributing factor
you find.

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
fully-asynchronous trainer on 8 GPUs (4 training, 4 serving), the pinned Qwen3.5-9B policy, the local
sandbox service, and three rollouts of the staged workload.

The run's shape is fixed and graded from the fresh run's own logs. These workspace files are
byte-compared against pristine copies and must not be modified: everything under `scripts/` and
`runtime/` other than `scripts/run_config.json` — including `runtime/sandbox_service.py`, the
sandbox provider itself, whose latencies and capacity are not yours to change. In
`scripts/run_config.json` only the sandbox client's own environment variables may change; the model,
topology, training and rollout sections are compared against the pristine configuration. Everything
else in the workspace, including `vendor/agent_runtime/` and the vendored framework source, may be
repaired.

Grading requires the run to complete three rollouts with a successful optimizer update each, and to
keep the workload intact: one sandbox per trajectory, every sandbox created, filled and installed
before use, every tool call executed in a sandbox, sandboxes deleted when their trajectory finishes,
and environment errors rare. It then measures the wall-clock time the first two rollouts take. Only
files from this fresh run are graded; do not copy, edit or fabricate metrics.
