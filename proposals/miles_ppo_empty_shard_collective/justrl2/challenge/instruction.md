# Distributed replay investigation

The supplied replay finishes its updates but does not match the expected validation results. Identify the cause and repair it while preserving the intended computation. The inputs are authored trajectories; critic values and policy log-ratios are synthetic. No model or optimizer runs.

Python 3.12+, CPU PyTorch/Gloo and NumPy are required. Run:

    python debug/run_window.py --output /tmp/run-a

Use a fresh output directory for each run. Submit source changes and report.md explaining the cause, correction and supporting validation. Preserve the inputs, objective and numerical semantics. Do not bypass operations or fabricate results. Work offline without GPU jobs. The evaluator checks the repair independently.
