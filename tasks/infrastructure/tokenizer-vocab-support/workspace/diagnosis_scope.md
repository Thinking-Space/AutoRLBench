# Diagnosis scope

Diagnose the systematic mismatch between rollout and recomputed actor log probabilities on the same generated token IDs in this recorded 7B continuation. Identify the implementation mechanism and evidence explaining the dominant shared log-probability shift; distinguish remaining outliers and other faults from this specific mismatch.

`experiment_logs/metrics.jsonl` contains every available scalar row from updates 726–1000. `experiment_logs/same_token_observations.json` contains original prompt/response token IDs and recorded rollout/actor token log probabilities for three selected long responses and one short response from the checkpoint-975 diagnostic replay. These are selected diagnostics, not a population sample. They preserve the exact observations, including outliers; source and selection are recorded in `experiment_logs/evidence_scope.json`.

The supplied framework and recorded configuration determine the active probability calculation paths. Trace the same actions through sampling and actor likelihood evaluation. Explain the systematic component of the disagreement and its optimization consequence. Report independent defects when supported, with a separate statement of which observations they explain. The requested diagnosis does not require an exhaustive causal account of the entire historical learning curve.

The scoring module is restored to its original immutable-runtime bytes. Historical rollouts outside the preserved diagnostic replay were not dumped. Do not fabricate missing responses or launch training.

`experiment_logs/shared_same_token_observations.json` contains a separate replay of one selected checkpoint-975 response in the benchmark shared environment. `experiment_logs/shared_raw_positions.json` records selected output logits and normalization statistics from the same checkpoint and response. Response positions are zero-based. The original continuation and the replay are distinct executions.
