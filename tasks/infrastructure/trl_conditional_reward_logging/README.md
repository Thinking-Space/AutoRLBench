# Distributed GRPO with conditional reward metrics

Qwen2.5-0.5B-Instruct is trained with GRPO on GSM8K. Process-local custom metric keys cause unmatched collectives when only one rank emits a metric.

| | |
| --- | --- |
| Author | Haoyang Su |
| Domain | infrastructure |
| Tags | `trl` `grpo` `gsm8k` |
| Model | Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K, 256 training questions and 64 held-out questions |
| Framework | TRL, `train.sh` |
| Environment | `trl-grpo-torch26`, 2 H100 80 GB GPU(s) in recorded runs |
| Evaluations | issue and solution |
| Recorded runs | One defective and one control run, seed 42, 24-step budget |

## Difficulty

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._generate_and_score_completions`: each rank iterates its own pending metric and extra-column keys before invoking collectives. A callback logs answer_parse_rate only on rank 0, so the ranks call different collectives at the same sequence number. Gather the union of keys first and iterate it identically on every rank. For metrics reduce the sum and count of actual supplied values, so an absent rank is not treated as a zero observation. The upstream repair also pads missing extra columns with None, but that unexercised path is not required for a passing diagnosis of this conditional-metric case.

The defective two-rank run raises a collective fingerprint mismatch before the first optimizer step. The control completes 24 optimizer steps and the final held-out evaluation, with reward 0.19140625. All 12 training answer_parse_rate values match the rank-0 rollout fractions, including 0.625 at the first generation event.

Reject diagnoses of GPU out-of-memory, inaccessible model weights, a generic network failure, bad arithmetic rewards, or insufficient model capacity. The collective mismatch is a communication-order consequence of process-local key sets. Disabling logging hides the defect and loses required diagnostics. This run directly exercises conditional metrics. The related extra-column fix is included from upstream, but no separate extra-column reproduction is claimed.

## Reference solution

The defective workspace is TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144. The control applies the conditional logging repair from upstream PR #7382, commit cc5bb3f8c94b7f9cf5c24029d7812e78849ccace. All experiment settings are identical between arms. The source and repair are available in [upstream PR #7382](https://github.com/huggingface/trl/pull/7382).
The issue tier requires the causal diagnosis. The solution tier includes a minimal patch and the control's actual run artifacts.
The original TRL Apache 2.0 license is retained in `workspace/LICENSE`. GSM8K is MIT licensed and the Qwen model is Apache 2.0 licensed.

## Verification

The issue judge scores verdict, root cause and evidence with weights 0.2, 0.6 and 0.2, passing at 0.8.
The solution protocol requires `report.md`, followed by a protected fresh training run that imports the agent's repaired trainer.
Both ranks must complete all 24 optimizer steps. The custom answer_parse_rate metric must remain present at all 12 generation events and equal the fraction of parsed answers on the rank that supplied it, within absolute and relative tolerance 0.000001. Disabling the metric or replacing an absent rank contribution with zero does not satisfy this criterion.

The score `synchronized_steps` is 0 in the defective run and 24 in the control. The threshold 23.5 has observed margins of 23.5 and 0.5.
The pre-training score is zero. Calibration uses one run per arm and does not estimate variance.
Both successful arms have nonzero gradient updates. Held-out reward is retained as context, with no claim of a statistically established accuracy improvement.

Native GPU execution and grader replay were tested. A second control run applied the packaged repair and passed the actual protected training and grading entry points. Both task preparation scripts also completed natively using the official asset helper with verified cached downloads. This repeat uses the same seed and is an end-to-end check, not a variance estimate. The Docker image is **not yet built**. Container execution and the LLM issue judge have not been exercised.
The recorded control takes about 4.2 minutes for training and held-out evaluation, excluding asset preparation. The verifier allows 30 minutes.
Preparation downloads approximately 2.6 MB of data and a 1 GB public model separately. Model weights and checkpoints are excluded from this task package.
