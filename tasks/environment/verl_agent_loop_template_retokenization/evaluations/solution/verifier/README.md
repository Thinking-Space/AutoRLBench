# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reads
`training/rollout_actor_probs_pearson_corr` from the console lines
(`step:N - ... training/rollout_actor_probs_pearson_corr:0.98...`) under
`$AUTORLBENCH_RESULTS/experiment_logs`. `tests/grade_submission.py` is the veRL variant
of the shared grader with one change: the value may be wrapped as `np.float64(...)`,
which is how verl 0.9 prints validation metrics.

## Why a mismatch metric and not accuracy

Two pairs of runs gave no held-out accuracy gap: at step 100 GSM8K is 89.1% vs 88.5%
and 86.5% vs 89.1% (defective vs control), GSM-Hard 67.8% vs 67.8%, and the same base
weights already differ by up to 3.4 points at step 0. No accuracy threshold can be
calibrated. The defect itself, the trainer optimising tokens the engine did not sample, 
is measured directly by verl every step: the Pearson correlation between rollout-engine
and trainer token probabilities over the response mask. It is near 0.99 whenever the
trainer sees the sampled tokens (bf16 vLLM vs FSDP noise only) and near 0 when the
sequence is rebuilt.

## Threshold

- `score_mode: final`, `final_step: 100`: the repaired run must complete the fixed budget.
- `required_final_score: 0.9`. Recorded step-100 values: defective 0.0071 and -0.0086,
  control 0.9887 and 0.9885. Over all 100 steps the defective runs never exceed 0.0433
  and the control runs never fall below 0.9865, so the bar is 0.857 above the highest
  defective value and 0.087 below the lowest control value.
- `required_final_score_max: 1.0` rejects impossible values.
- Calibrated on two runs per side (one per pair). Margins only, no variance estimate.

Verified against the recorded logs: both defective runs score 0 (`score 0.0071 at step
100 is below required score 0.9`), both control runs score 1.

## Known limits

- A change that keeps the re-rendered sequence but obtains matching log-probs for it
  (for example by asking vLLM for prompt log-probs of the rebuilt ids) would pass while
  still training on tokens that were never sampled. The issue tier grades the attribution.
- A run with `calculate_log_probs=False` logs no correlation and scores 0. The instruction
  fixes that setting.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`.
`test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.
