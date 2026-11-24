# Solution verifier

The stock `test.sh` runs `tests/grade_submission.py`. It checks the protected experiment files listed in `resources/grader_spec.json` and reads `val-core/overall/acc/mean@32` from the fresh `experiment_logs` at step 1000. `scripts/emit_heldout_score.py` rescales no metrics. It independently scores the native validation dump, checks prompt and answer identities, and emits the combined fraction correct after the rerun verifies the native CP990 and CP1000 checkpoints.

The score must reach 2,162 correct answers out of 3,200 (0.675625). The stock verifier rejected the fault result (2,012 correct) and accepted the repair result (2,173 correct). The calibration uses one segmented native training run per arm.
