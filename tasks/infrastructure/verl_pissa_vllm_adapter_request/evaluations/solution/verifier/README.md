# Solution verifier

`test.sh` parses only fresh text logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. It requires finite
`val-health/actor_rollout_policy_agreement >= 0.80` and `<= 1.0` at exactly
step 4. The recorded defective/control values are `0.10476375009925555` and
`0.8907221660212565`. Calibration uses one run per side, so the margins are
empirical rather than variance bounds. No reward-shaping metric is graded.
