# Grading resources

`../test.sh` is the entry point. The fresh run is `python3 launch.py`, with the
environment and 3600-second limit in `run_spec.json`. The verifier reads
`summary.json` and `experiment_metrics.json` under
`$AUTORLBENCH_RESULTS/experiment_logs`.

`grader_spec.json` records the final horizon of 16 and the minimum of 12
error-free episodes. Tool-history retention must be in [0.99, 1.0], agree with
the raw counters, and pass the completion adapter's hidden compatibility checks.
