# verl_rollout_weight_overwrite

Recorded veRL GRPO case, with a SHA-pinned reproducible data preparation stage and issue/solution evaluation protocol. Source readiness: validated; container build and GPU acceptance remain deferred at the user's request.

| Field | Value |
|---|---|
| Domain | infrastructure |
| Model | Qwen/Qwen3-1.7B @70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, no thinking |
| Data | DAPO original1,791,700 rows (no dedup); AIME24/25 30 questions each |
| Framework | veRL0.6.1 with recorded compatibility patches; torch2.8.0, vLLM0.11.0, FA2.8.3, Transformers4.57.1 |
| Image | verl-v061-grpo; Dockerfile registered, **not yet built** |
| Recorded topology | Two physical nodes x4 GPUs |
| Portable topology | Two logical Ray nodes x4 GPUs inside one8-GPU container, or external two-node Ray cluster |
| Budget | 150 steps; batch512, mini32, rollout8; validation0/10/.../150 mean@32 |
| Independent runs | One candidate and ONE shared task2 reference; not seven independent repaired replicas |
| Levels | issue and solution |

## Difficulty
Connect the concrete source mechanism to recorded observations. Low accuracy or unusual configuration alone is insufficient attribution.
The staged workspace includes a declared benchmark target contract, added during packaging, to distinguish comparison against standard GRPO from universal claims about alternative algorithms.

## Reference solution
The filtered code/config repair is in evaluations/solution/oracle/fix.patch, and the restored effective recipe is in restored_config.yaml. Native unpadding interpolation is preserved. All proposed patches apply cleanly. task2 repaired run sbr1bh7d is the unique shared reference; it lives outside agent-visible workspaces under shared_baselines/task2_repaired_grpo. Task4 has no independently recorded repaired synchronization trace. The three non-configuration source mechanisms are controlled ports/injections, not claims of native upstream veRL defects.

## Verification
The official issue scaffold is copied byte-for-byte from upstream259ebf12c2f2b78c60f7500e124115c0aaeae4a5. The solution agent edits code and submits /workspace/report.md without training. The harness clears experiment_logs and runs bash run.sh. Budget, protocol and output roots are identical in instruction.md/run_spec.json.

Score metric: aime24_25_mean32_average. Candidate=0.012500000000; shared reference=0.143750000000; threshold=0.07. Observed margins: candidate-to-threshold=0.057500000000; reference-to-threshold=0.073750000000. Calibration uses ONE run per side and reports margins, not variance. The step150 requirement prevents a step0-only submission. Scoring uses complete ordered metrics, all16 scheduled evals, independent final answer rescoring, immutable data hashes, the declared recipe, and final checkpoint archive/header checks. Task1 also independently re-scores all training rollouts121..150. No historical logs are graded during the normal protocol.

Data preparation was executed natively: both parquet outputs matched recorded SHA256, size and row counts exactly. Cold model/data acquisition, cached reuse and corrupt/partial output rejection were checked. Fourteen candidate/restored Hydra configurations composed without training; every restored configuration matched the protected contract. Seven strict reference verifier replays passed on original saved evidence; this is NOT a new GPU training run. Seven candidate historical replays failed their thresholds. Detailed receipts live under submission_validation/ at repository root.

## Remaining acceptance checks
Container build, eight-GPU partitioning and fresh container/GPU reruns have NOT been performed. The historical run used two physical nodes; the one-host/two-logical-node portability mode has only CPU Ray-node isolation validation. Use an external two-node Ray cluster to preserve physical topology. Live LLM diagnosis judging has not been run. A current upstream generic GPU-stage unit test has a known prepare-stage expectation mismatch; see the preserved regression note. Benchmark acceptance and seven independently countable contributions remain maintenance decisions.
