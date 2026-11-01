# Solution verifier

`test.sh` parses only fresh text logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. It requires finite
`val-health/base_reference_consistency >= 0.95` and `<= 1.0` at exactly step
4. The recorded defective/control values are `0.045778327704709` and `1.0`.
Calibration uses one run per side, so the margins are empirical rather than
variance bounds. The task does not claim a measured step-zero value.
