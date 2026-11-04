# Solution verifier

`test.sh` parses only fresh text logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. It requires finite
`val-health/actor_rollout_policy_agreement >= 0.80` and `<= 1.0` at exactly
step 4. The recorded defective/control values are `0.10476375009925555` and
`0.8907221660212565`. Calibration uses one run per side, so the margins are
empirical rather than variance bounds. No reward-shaping metric is graded.

Because the graded metric is computed by instrumentation that lives in the
shipped workspace, the value alone proves nothing. The grader anchors it four
ways:

1. **Run reality (every grading).** verl's file logger must show one
   `{"step", "data"}` row per step 0..4 with the runner's per-step metrics,
   the final row must carry the mae/agreement pair with
   `agreement == exp(-mae)`, and the value the text log was judged on must
   equal the file-log value at step 4. The runner's own artifacts
   (`source_commit.txt`, `environment-freeze.txt`, `nvidia-smi.txt`) and the
   rollout/validation dumps must be present.
2. **Behavioral anchor (every grading).** The rollout sync probe dump must
   still show the PiSSA actor (`is_lora`, `is_svd_init`, `rollout_sync_mode`
   `lora_delta`, the PiSSA adapter path), and a majority of the per-request
   generate probe dumps must show `lora_request_used`. This rejects the
   masked repair of disabling the adapter everywhere, which would agree at
   ~1.0 while training the residual base only.
3. **Stage integrity (live grading only).** With the stage environment set,
   `run_experiment.sh` and `verl/verl/trainer/ppo/ray_trainer.py` are
   sha256-pinned against `grader_spec.json`, so editing the budget, the
   launch script or the health computation fails. The repair itself lives in
   the rollout request path and is not pinned.
4. **Replay contract.** A hand replay of packaged evidence runs without the
   stage environment and skips check 3.

Not defended (family-wide residual): a post-hoc forgery that fabricates a
self-consistent file log, probe dumps and artifacts after tampering with
unpinned instrumentation and restoring it. The GPU qualification run should
confirm the grader accepts a real repaired run end-to-end.
