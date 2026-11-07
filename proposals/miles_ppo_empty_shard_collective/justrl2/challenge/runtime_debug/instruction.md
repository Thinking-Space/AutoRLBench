# Intermittent replay failure

A CPU replay of a distributed PPO update sometimes completes and sometimes
stalls, depending on the workload and process layout. Investigate the source
and run bounded experiments to locate the initiating defect. Preserve the
intended GAE, returns, loss masks and pooled normalization semantics.

The workspace contains the native framework and `debug/replay_probe.py`. The
probe executes its CP slicing, GAE and normalization functions on real Gloo
groups with synthetic critic values; it does not run a model or GPU training.
Python 3.12+ and CPU PyTorch with NumPy are required. A starting run is:

```bash
python debug/replay_probe.py --output /tmp/probe-a --cp 2 --dp 2 --steps 11,3
```

Use a fresh output directory for each run. `--help` lists workload, rank-layout
and tracing controls. Each command starts fresh processes; successive steps in
one command share process state. You may edit and instrument the framework,
and use the provided trace option. Do not use the network or launch GPU jobs.

Write `report.md` with the following debugging evidence:

1. Record the commands and results for a failure and at least two meaningful
   passing controls. Vary workload, layout or ordering to distinguish hypotheses.
2. Identify the first operation whose behavior disagrees with its caller's
   intent. Show rank-level evidence and explain why later errors are secondary.
3. Give a causal explanation spanning code and runtime state, including which
   conditions trigger the failure and which plausible explanations are ruled out.
4. Implement the smallest justified correction. Re-run the failure, passing
   controls and a previously untested layout/input; preserve numerical outputs.

Include actual run paths and source hashes from the probe outputs so the
evaluator can correlate the report with fresh executions. Static suspicions or
a patch without runs are incomplete. Do not fabricate traces or edit probe
results; do not change inputs merely to avoid a failing condition. The evaluator
uses its own trusted runner and independent numerical checks after submission.

Also write `experiments.json` with `report` pointing to `report.md`, and `before`
and `after` listing the corresponding probe output directories. Preserve all
raw artifacts. Include two failures whose first mismatches differ with execution
history and at least two distinct passing controls. After repair, repeat every
listed experiment and add at least one previously untested configuration. The
evaluator checks source hashes, actual collective traces and numerical results;
a repair without these debugging artifacts is incomplete.
