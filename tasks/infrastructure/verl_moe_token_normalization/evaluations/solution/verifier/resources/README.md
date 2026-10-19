# Scoring resources

Entry point: `../test.sh`. Only fresh files in
`$AUTORLBENCH_RESULTS/experiment_logs` are scored. The target is
`val-core/openai/gsm8k/acc/mean@1` at step 150, with `score_mode=final`.
Pass requires accuracy >= 0.4 and <= 1.0. Historical defective baseline is 0.
No auxiliary metric is weighted. The accuracy key is a fraction, not percent.
