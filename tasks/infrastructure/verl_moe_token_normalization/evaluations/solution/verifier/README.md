# Solution verifier

Run `test.sh` with `AUTORLBENCH_RESULTS` pointing to fresh run results.
The task-local grader reuses the repository's metric grader with one change:
step/metric matching is bounded to a single physical veRL console line, so
step 100 or 150 validation cannot attach to a preceding training step.

`val-core/openai/gsm8k/acc/mean@1` is accuracy on the prepared 64-question
held-out set. Passing requires a finite value >= 0.4 and <= 1.0 at exactly
step 150. It writes experiment_metrics.json, metrics.json and result.json.
No judge or native-gradient probe determines this full-training score.

Calibration uses 1 run per arm: defective 0/0 and repaired
0.453125/0.546875 at steps 100/150. Final margins to threshold are 0.4 and
0.146875; the smallest margin across observed repaired points is 0.053125.
Cross-seed variance and step-0 validation are unavailable. A metric at step
0 or an earlier best step does not satisfy the final-step requirement.
