# verl_rollout_weight_overwrite

Recorded veRL GRPO case, with a SHA-pinned reproducible data preparation stage and issue/solution evaluation protocol. Source readiness: validated; container build and GPU acceptance remain deferred at the user's request.

| | |
| --- | --- |
| **Author** | Pengyi Li |
| **Affiliation** | FusionBrain Lab |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `initial-checkpoint-reload` `weight-sync-overwrite` `stale-rollout-policy` |
| **Framework** | verl 0.6.1 (GRPO), torch 2.8.0 / vLLM 0.11.0 / FA 2.8.3 / Transformers 4.57.1, with recorded compatibility patches |
| **Run time** | ≤ 4 days |
| **Resources** | 8 GPU |

## Difficulty
Connect the concrete source mechanism to recorded observations. Low accuracy or unusual configuration alone is insufficient attribution.
The staged workspace includes a declared benchmark target contract, added during packaging, to distinguish comparison against standard GRPO from universal claims about alternative algorithms.

## Reference solution
- **issue**: the expected verdict is unhealthy, and the root cause is `workspace/case/sync.py` `finalize_weights` reloading the initial safetensors checkpoint into the vLLM engine after Actor weights have been synchronized — the post-sync hook replaces the just-synced weights with the initial checkpoint, so rollout generation repeatedly uses the initial weights while the Actor keeps training against its own updated weights (rollout engine reset to initial weights → generation never uses the updated Actor → training signal computed against stale samples → flat reward and frozen held-out accuracy). Naming the post-sync initial-checkpoint reload with its correct causal role is a complete attribution; "zero gradients / no learning" is wrong (`actor/grad_norm` grows 0.24 → max 83.7) and "the reward callback broadcasts one score per batch" is wrong (advantages are nonzero).
- **solution**: the filtered code/config repair is in evaluations/solution/oracle/fix.patch, and the restored effective recipe is in restored_config.yaml. The post-sync hook becomes telemetry-only — it records the fingerprint diff (`sync_trace.jsonl`) without loading any checkpoint. Native unpadding interpolation is preserved. All proposed patches apply cleanly. task2 repaired run sbr1bh7d is the unique shared reference; it lives outside agent-visible workspaces under shared_baselines/task2_repaired_grpo. Task4 has no independently recorded repaired synchronization trace. The three non-configuration source mechanisms are controlled ports/injections, not claims of native upstream veRL defects.

## Verification
- **issue**: the official issue scaffold is copied byte-for-byte from upstream259ebf12c2f2b78c60f7500e124115c0aaeae4a5. The agent writes a diagnosis without launching training, and `agent_answer.md` is graded against the hidden oracle with an LLM judge scoring verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2; passing requires verdict score exactly 1.0 and both root-cause and evidence scores above 0.8.
- **solution**: the solution agent edits code and submits /workspace/report.md without training. The harness clears experiment_logs and runs bash run.sh. Budget, protocol and output roots are identical in instruction.md/run_spec.json.

Score metric: aime24_25_mean32_average. Candidate=0.012500000000; shared reference=0.143750000000; threshold=0.07. Observed margins: candidate-to-threshold=0.057500000000; reference-to-threshold=0.073750000000. Calibration uses ONE run per side and reports margins, not variance. The step150 requirement prevents a step0-only submission. Scoring uses complete ordered metrics, all16 scheduled evals, independent final answer rescoring, immutable data hashes, the declared recipe, and final checkpoint archive/header checks. Task1 also independently re-scores all training rollouts121..150. No historical logs are graded during the normal protocol.

Data preparation was executed natively: both parquet outputs matched recorded SHA256, size and row counts exactly. Cold model/data acquisition, cached reuse and corrupt/partial output rejection were checked. Fourteen candidate/restored Hydra configurations composed without training; every restored configuration matched the protected contract. Seven strict reference verifier replays passed on original saved evidence; this is NOT a new GPU training run. Seven candidate historical replays failed their thresholds. Detailed receipts live under submission_validation/ at repository root.

## Remaining acceptance checks
Container build, eight-GPU partitioning and fresh container/GPU reruns have NOT been performed. The historical run used two physical nodes; the one-host/two-logical-node portability mode has only CPU Ray-node isolation validation. Use an external two-node Ray cluster to preserve physical topology. Live LLM diagnosis judging has not been run. A current upstream generic GPU-stage unit test has a known prepare-stage expectation mismatch; see the preserved regression note. Benchmark acceptance and seven independently countable contributions remain maintenance decisions.
