# Hidden solution evidence

`baseline/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` is the control-side file copied from the source snapshot. `control_train.log` and `control_step0.log` are sanitized copies of the baseline 50-step and step-0 evaluations. The threshold evidence is summarized in `historical_evidence_manifest.json`.

Only one run per arm was recorded. The threshold is therefore a reproducibility check for this observed A/B pair, not a claim about variance or statistical significance.
