# verl_policy_loss_padded_mask_only

Recorded veRL GRPO case, with a SHA-pinned reproducible data preparation stage and issue/solution evaluation protocol. Source readiness: validated; container build and GPU acceptance remain deferred at the user's request.

| | |
| --- | --- |
| **Author** | Pengyi Li |
| **Affiliation** | FusionBrain Lab |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `policy-loss` `objective-aggregation` `padded-tokens` `mask-repair` |
| **Framework** | verl 0.6.1 (GRPO), torch 2.8.0 / vLLM 0.11.0 / FA 2.8.3 / Transformers 4.57.1, with recorded compatibility patches |
| **Run time** | ~64.5 hours |
| **Resources** | 8 GPU |

## Difficulty
This case restores a declared standard-GRPO target. Sequence-sum objectives are not universally wrong. The original mask-only runs retain their original role in reviewer metadata. Restoring the chosen target changes normalization, clipping, and (for retained-padding cases) forward mode; this is not a single-mask causal intervention.
The staged workspace includes a declared benchmark target contract, added during packaging, to distinguish comparison against standard GRPO from universal claims about alternative algorithms.

## Reference solution
- **issue**: the expected verdict is unhealthy, and the root cause is composite — the custom `luspo_token` policy loss with `seq-mean-token-sum` aggregation in `workspace/case/loss.py` (summing valid token losses per response instead of the declared standard token-mean aggregation, and omitting native dual clipping), together with `use_remove_padding: false` in `workspace/case/recorded_config.yaml`. Naming only the custom loss or only the disabled unpadding is partial attribution; naming both with their correct causal roles is complete.
- **solution**: the filtered code/config repair is in evaluations/solution/oracle/fix.patch, and the restored effective recipe is in restored_config.yaml. Native unpadding interpolation is preserved. All proposed patches apply cleanly. task2 repaired run sbr1bh7d is the unique shared reference; it lives outside agent-visible workspaces under shared_baselines/task2_repaired_grpo. Task4 has no independently recorded repaired synchronization trace. The three non-configuration source mechanisms are controlled ports/injections, not claims of native upstream veRL defects.

## Verification
- **issue**: the official issue scaffold is copied byte-for-byte from upstream259ebf12c2f2b78c60f7500e124115c0aaeae4a5. The agent writes a diagnosis without launching training, and `agent_answer.md` is graded against the hidden oracle with an LLM judge scoring verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2; passing requires verdict score exactly 1.0 and both root-cause and evidence scores above 0.8.
- **solution**: the solution agent edits code and submits /workspace/report.md without training. The harness clears experiment_logs and runs bash run.sh. Budget, protocol and output roots are identical in instruction.md/run_spec.json.

Score metric: aime24_25_mean32_average. Candidate=0.057291666667; shared reference=0.143750000000; threshold=0.1. Observed margins: candidate-to-threshold=0.042708333333; reference-to-threshold=0.043750000000. Calibration uses ONE run per side and reports margins, not variance. The step150 requirement prevents a step0-only submission. Scoring uses complete ordered metrics, all16 scheduled evals, independent final answer rescoring, immutable data hashes, the declared recipe, and final checkpoint archive/header checks. Task1 also independently re-scores all training rollouts121..150. No historical logs are graded during the normal protocol.

Data preparation was executed natively: both parquet outputs matched recorded SHA256, size and row counts exactly. Cold model/data acquisition, cached reuse and corrupt/partial output rejection were checked. Fourteen candidate/restored Hydra configurations composed without training; every restored configuration matched the protected contract. Seven strict reference verifier replays passed on original saved evidence; this is NOT a new GPU training run. Seven candidate historical replays failed their thresholds. Detailed receipts live under submission_validation/ at repository root.

## Remaining acceptance checks
Container build, eight-GPU partitioning and fresh container/GPU reruns have NOT been performed. The historical run used two physical nodes; the one-host/two-logical-node portability mode has only CPU Ray-node isolation validation. Use an external two-node Ray cluster to preserve physical topology. Live LLM diagnosis judging has not been run. A current upstream generic GPU-stage unit test has a known prepare-stage expectation mismatch; see the preserved regression note. Benchmark acceptance and seven independently countable contributions remain maintenance decisions.
