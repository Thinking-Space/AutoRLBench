# Solution verifier

`test.sh` parses only fresh files under
`$AUTORLBENCH_RESULTS/experiment_logs`. It requires finite
`val-health/base_reference_consistency >= 0.95` and `<= 1.0` at exactly step
4, and anchors the run's integrity four ways beyond the metric bar:

1. verl's per-step file log (`metrics/all_steps.jsonl`) must show a complete
   run (steps 0..4), its final row must carry the
   `base_reference_mae`/`base_reference_consistency` pair with
   `consistency == exp(-mae)`, and the judged text-log value must equal the
   file-log value.
2. every training step must keep `actor/kl_loss` under 0.1: with the
   reference correctly rebuilt from the initial policy the recorded repaired
   run logs 0.0 at all four updates, while the defective reference pulls it
   to 0.7618-0.9751. This rejects masked repairs (for example re-running
   with `lora_residual_base=False`, which vacuously compares the wrong
   reference against itself) even when the health pair itself clears the bar.
3. the rollout sync probe dump must still show the PiSSA actor (`is_lora`,
   `is_svd_init`, `rollout_sync_mode=lora_delta`, the PiSSA adapter path).
4. live grading sha256-pins the runner (`run_experiment.sh`) and the health
   instrumentation (`verl/verl/trainer/ppo/ray_trainer.py`) against
   `grader_spec.json`, so editing the budget, the launch flags or the metric
   computation fails; the repair itself lives in the worker's base-reference
   path and is not pinned. A hand replay of packaged evidence runs without
   the stage environment and skips the workspace checks.

The recorded defective/control values are `0.045778327704709` and `1.0`.
Calibration uses one run per side, so the margins are empirical rather than
variance bounds. The task does not claim a measured step-zero value.
