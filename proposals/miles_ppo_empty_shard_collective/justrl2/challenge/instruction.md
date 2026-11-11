# Distributed update investigation

The supplied update replay does not finish successfully under the starting
configuration. Identify the cause and repair it while preserving the intended
computation. The inputs are authored trajectories; the CPU replay uses
synthetic critic values and runs no model or optimizer.

Python 3.12+, CPU PyTorch/Gloo and NumPy are required. The starting command is:

```bash
python debug/run_updates.py --output /tmp/run-a
```

Use a fresh output directory for each run. Submit your source changes and
`report.md`, explaining the cause, the correction and supporting validation.

Preserve the replay inputs, objective and numerical semantics. Do not bypass
training operations to avoid the failure or fabricate results. Work offline
without GPU jobs. The evaluator checks the repair independently.
