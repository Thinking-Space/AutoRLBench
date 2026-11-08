# Distributed replay investigation

The supplied replay fails for some workloads. Determine the initiating defect,
repair it, and preserve the intended numerical computation. This is a CPU
function replay with synthetic values; it does not train a model.

Use Python 3.12+, CPU PyTorch/Gloo and NumPy:

```bash
python debug/run_replay.py --cp 2 --dp 2 --steps 11,3 --output /tmp/run-a
```

Each command starts fresh processes. Steps within a command share state. Use
fresh output directories and bounded changes to the workload or layout as
controls. The initial runner reports aggregate completion only. It contains no
tracing switch, tensor snapshots, route event stream or diagnostic hooks.

Add your own source logging or assertions to observe relevant runtime state.
Worker stdout/stderr captures what you add. Preserve an instrumentation-only
diff and its raw output before implementing a repair. Choose what to observe;
no fixed logging locations or schema are supplied.

Submit `report.md`, instrumentation and repair diffs, exact commands and raw
results. Reproduce failure, gather your own evidence, distinguish plausible
causes using passing controls, then repair and repeat the original case and a
new configuration. Explain the earliest observed divergence and the purpose of
each added observation. A static guess or patch without collected debugging
evidence is incomplete.

Keep the objective and normalization semantics. Do not change inputs merely to
avoid a failing condition, fabricate traces or edit saved outputs. Do not use
the network or GPU jobs. Correctness is checked separately with a trusted
runner, unseen cases and numerical comparisons.
