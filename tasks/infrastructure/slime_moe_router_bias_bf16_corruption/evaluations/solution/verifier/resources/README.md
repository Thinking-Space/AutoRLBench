# Solution verifier resources

`../test.sh` runs `tests/grade_submission.py`. It reads only fresh JSONL files under `$AUTORLBENCH_RESULTS/experiment_logs`, requires the declared step-12 probability and router invariants, checks a nonempty workspace `report.md`, and verifies the packaged `train.py` SHA-256.
