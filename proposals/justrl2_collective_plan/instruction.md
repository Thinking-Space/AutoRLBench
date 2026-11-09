# Distributed replay investigation

The supplied replay fails for some workloads. Identify the cause and repair
it while preserving the intended numerical computation. This CPU function
replay uses synthetic values and does not train a model.

Python 3.12+, CPU PyTorch/Gloo and NumPy are required. The starting command is:

```bash
python debug/run_replay.py --cp 2 --dp 2 --steps 11,3 --output /tmp/run-a
```

Use a fresh output directory for each run. Submit your source changes and
`report.md`, explaining the cause, the correction and supporting validation.

Preserve the objective and numerical semantics. Do not change inputs merely
to avoid a failing condition or fabricate results. Work offline without GPU
jobs. The evaluator checks the repair independently.
