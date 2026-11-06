# Solution grading resources

Entry point: `../test.sh`. The grader reads only fresh text files below `$AUTORLBENCH_RESULTS/experiment_logs` and writes `experiment_metrics.json`, `metrics.json` and `result.json`.

`grader_spec.json` uses schema version 2, `metric_key=critic/score/mean`, `final_step=200`, `score_mode=final`, historical baseline -1.0, passing threshold 0.5, and maximum 1.0. Historical workspace evidence is not a submission. A missing final point, non-finite value, or out-of-range score fails. This criterion measures recovery from training collapse.

The spec additionally carries the anchoring layers of the hardened grader:

- `min_step_coverage` (0.95) — the log the run is judged from must show this fraction of the training steps 1..200 for the graded metric;
- `required_log_patterns` — configuration anchors that must be visible in the judged log (staged datasets, model, GRPO group size, tool-service launches, launcher wiring);
- `corroboration_metrics` — step-200 bounds checked in the same log (`verl_tool/is_traj_finished/mean` ≥ 0.9, `response_length/clip_ratio` ≤ 0.05);
- `pinned_sha256` — live-only sha256 pins of the launch/logging chain (runner, launcher, console logger); checked when the stage sets `AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_VERIFIER`, skipped by hand replays.
