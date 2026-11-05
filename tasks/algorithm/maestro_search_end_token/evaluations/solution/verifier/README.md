# Solution verifier

The unchanged common `test.sh` invokes the standard schema-v2 `tests/grade_submission.py`. `score_mode=final` requires the logged `critic/score/mean` at step 200 rather than accepting the defective run's temporary earlier improvement. The captured step-120 defective score is 0.7109375, which would pass an inappropriate best-of-run 0.5 threshold.

The threshold is 0.5, calibrated against one defective final (-1.0) and one control final (0.6796875); margins are 1.5 and 0.1796875. Maximum 1.0 is the natural upper bound of this reward. No step-0 training score or multi-seed variance was recorded. The criterion measures recovery from the training collapse documented by the reward curve and rollout termination behavior.

Only new files under `$AUTORLBENCH_RESULTS/experiment_logs` count. The verifier returns a binary score and the usual `experiment_metrics.json`, `metrics.json`, and `result.json`. The contributor must not submit copied or fabricated metric logs.
