# Distributed update investigation

The supplied update replay does not finish successfully under the starting
configuration. Locate and repair the initiating defect while preserving the
intended computation. The inputs are authored trajectories; the CPU replay
uses synthetic critic values and runs no model or optimizer.

Start with Python 3.12+, CPU PyTorch/Gloo and NumPy:

```bash
python debug/run_replay.py --replay inputs/replay.json --cp 2 --dp 2 --output /tmp/run-a
```

Use a fresh output directory each time. The runner reports aggregate completion
only. It supplies no diagnostic event stream, saved tensors, group ledger or
trace switch. Native framework copies contain no supplied print/logger calls.
You may add your own logging or assertions in source to observe runtime state;
worker stdout/stderr is captured for that purpose. Keep an instrumentation-only
diff separate from the repair so evidence can be attributed to the right version.

Submit `report.md`, your source changes, the instrumentation-only diff, commands
and raw outputs. First reproduce the failure, then use your own observations
and controlled experiments to distinguish causes. Explain the earliest
divergence, repair it, and rerun the original case plus a relevant control.
Record what each added observation measures and why it supports the diagnosis.
A static guess or a repair without collected debugging evidence is incomplete.

Do not change replay inputs or bypass the objective merely to avoid the error.
Do not fabricate logs or edit recorded outputs. Do not use the network or run
GPU training. An external evaluator uses independent checks for correctness.
