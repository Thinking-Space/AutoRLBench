# Distributed GRPO with conditional reward metrics

Qwen2.5-0.5B-Instruct is trained with GRPO on GSM8K. Process-local custom metric keys cause unmatched collectives when only one rank emits a metric.

| | |
| --- | --- |
| **Author** | Haoyang Su |
| **Affiliation** | Fudan University; Shanghai Innovation Institute |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `collective-mismatch` `rank-divergent-metrics` `conditional-logging` |
| **Framework** | TRL (GRPO) |
| **Run time** | ~4.2 minutes per run |
| **Resources** | 2 GPU |

## Difficulty

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._generate_and_score_completions`: each rank iterates its own pending metric and extra-column keys before invoking collectives. The reward function logs answer_parse_rate only on rank 0, so the ranks call different collectives at the same sequence number. Gather the union of keys first and iterate it identically on every rank. For metrics reduce the sum and count of actual supplied values, so an absent rank is not treated as a zero observation.

The defective two-rank run raises a collective fingerprint mismatch before the first optimizer step. The control completes 24 optimizer steps and the final held-out evaluation, with reward 0.19140625. All 12 training answer_parse_rate values on both ranks match the rank-0 rollout fractions, including 0.625 at the first generation event.

Reject diagnoses of GPU out-of-memory, inaccessible model weights, a generic network failure, bad arithmetic rewards, or insufficient model capacity. The collective mismatch is a communication-order consequence of process-local key sets. Disabling logging hides the defect and loses required diagnostics. This run exercises conditional metrics.

## Reference solution

- **issue**: The issue tier requires the causal diagnosis.
- **solution**: The defective workspace is TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144. The control applies the conditional logging repair from upstream PR #7382, commit cc5bb3f8c94b7f9cf5c24029d7812e78849ccace. All experiment settings are identical between arms. The source and repair are available in [upstream PR #7382](https://github.com/huggingface/trl/pull/7382). The solution tier includes a minimal patch and the control's actual run artifacts.

The original TRL Apache 2.0 license is retained in `workspace/LICENSE`. GSM8K is MIT licensed and the Qwen model is Apache 2.0 licensed.

## Verification

- **issue**: The issue judge reports a weighted score for verdict, root cause and evidence, with weights 0.2, 0.6 and 0.2. Passing requires a verdict score of exactly 1.0 and both root-cause and evidence scores strictly greater than 0.8.
- **solution**: The solution protocol requires `report.md`, followed by a protected fresh training run: the system runs the workspace's `run_training.sh`, which byte-matches the verifier's pristine copy, and grading reads both ranks' metric streams from the run. Both ranks must complete all 24 optimizer steps. The custom answer_parse_rate metric must remain present at all 12 generation events on both ranks, on the same steps, and equal the fraction of parsed answers in this run's rank-0 rollouts at the preceding step, within absolute and relative tolerance 0.000001. Disabling the metric, logging it on rank 0 only, replacing an absent rank contribution with zero, or concentrating the rank-1 events on a subset of the 12 steps does not satisfy this criterion.

The score `synchronized_steps` is 0 in the defective run and 24 in the control. The threshold 23.5 has observed margins of 23.5 and 0.5.
The pre-training score is zero. Calibration uses one run per arm with seed 42.
Successful control runs have nonzero gradient updates. Held-out reward provides a secondary descriptive measure.

Native GPU execution and task preparation passed. The packaged rollout streams were thinned for size (three exemplar rows per stream), so the offline grader replay of the packaged control does not pass as shipped: the metric check needs the full per-event rollout rows to recompute the expected fractions. Restoring the control arm's complete rollout rows is the remaining pre-merge item. Container validation remains pending.

The bundled issue judge was exercised with Qwen3-4B in bfloat16 on an H100 using Transformers 5.15.1. A correct diagnosis passed with score 1.0, while an incorrect causal explanation and an incorrect healthy verdict each received score 0.0. Run `evaluations/issue/verifier/judge.py` with an input, the task oracle and judge prompt to repeat the procedure.
The recorded control takes about 4.2 minutes for training and held-out evaluation, excluding asset preparation. The verifier allows 30 minutes.
Preparation downloads approximately 2.6 MB of data and a 1 GB public model separately.

The offline W&B streams for both recorded arms are included beside their text logs. They contain the original metric histories and summaries, with host names standardized; recording-host paths are listed in `environment/source_manifest.json`.
