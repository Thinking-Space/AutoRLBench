# Solution grading resources

Entry point: `../test.sh`. The grader reads only fresh text files below `$AUTORLBENCH_RESULTS/experiment_logs` and writes `experiment_metrics.json`, `metrics.json` and `result.json`.

`grader_spec.json` uses schema version 2, `metric_key=critic/score/mean`, `final_step=200`, `score_mode=final`, historical baseline -1.0, passing threshold 0.5, and maximum 1.0. Historical workspace evidence is not a submission. A missing final point, non-finite value, or out-of-range score fails. This criterion measures recovery from training collapse.
