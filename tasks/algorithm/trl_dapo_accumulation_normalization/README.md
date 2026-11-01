# GRPO generation and accumulation windows

Qwen2.5-0.5B-Instruct is trained with GRPO on GSM8K. The generation token denominator is not scaled to the accumulation window, giving half the intended loss.

| | |
| --- | --- |
| Author | Haoyang Su |
| Domain | algorithm |
| Tags | `trl` `grpo` `gsm8k` |
| Model | Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K, 256 training questions and 64 held-out questions |
| Framework | TRL, `train.sh` |
| Environment | `trl-grpo-torch26`, 1 H100 80 GB GPU(s) in recorded runs |
| Evaluations | issue and solution |
| Recorded runs | One defective and one control run, seed 42, 24-step budget |

## Difficulty

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._compute_loss`: the DAPO denominator uses the token count of the entire generation batch without scaling it to the current gradient accumulation window. With steps_per_generation=4 and gradient_accumulation_steps=2, the loss and raw gradient for identical inputs are half the intended accumulation-window objective. During training multiply the denominator by current_gradient_accumulation_steps / steps_per_generation. Evaluation uses its unsplit batch denominator.

Both arms complete 24 optimizer steps. On nonzero captured objectives, the defective loss/reference ratio is approximately 0.5 and the control ratio is approximately 1.0. The independent objective comparison passes 12/24 optimizer steps in the defective run and 24/24 in the control. The held-out rewards are 0.1953125 and 0.20703125 respectively.

Reject diagnoses of no learning, corrupted GSM8K labels, an incorrect numerical-answer reward, insufficient model size, or a guaranteed twofold Adam parameter update. Adam adaptation and gradient clipping can compensate for loss scaling. The exact objective discrepancy is the primary endpoint.

## Reference solution

The reproduction starts from TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144 and removes only the training denominator correction corresponding to upstream PR #6024. The task reproduces this documented historical defect. The source and repair are available in [upstream PR #6024](https://github.com/huggingface/trl/pull/6024).
The issue tier requires the causal diagnosis. The solution tier includes a minimal patch and the control's actual run artifacts.
The original TRL Apache 2.0 license is retained in `workspace/LICENSE`. GSM8K is MIT licensed and the Qwen model is Apache 2.0 licensed.

## Verification

The issue judge reports a weighted score for verdict, root cause and evidence, with weights 0.2, 0.6 and 0.2. Passing requires a verdict score of exactly 1.0 and both root-cause and evidence scores strictly greater than 0.8.
The solution protocol requires `report.md`, followed by a protected fresh training run that imports the agent's repaired trainer.
For every optimizer step, the verifier recomputes the DAPO objective from the recorded per-token probabilities, advantages and completion masks. Both microbatches must agree with the accumulation-window objective within relative tolerance 0.0005 and absolute tolerance 0.000002. All 24 optimizer steps must pass, with 48 records and at least one nonzero objective comparison.

The score `correct_objective_steps` is 12 in the defective run and 24 in the control. The threshold 23.5 has observed margins of 11.5 and 0.5.
The pre-training score is zero. Calibration uses one run per arm with seed 42.
Successful control runs have nonzero gradient updates. Held-out reward provides a secondary descriptive measure.

Native GPU execution, task preparation and grader replay passed. A second control run applied the packaged repair and passed the training and grading entry points with the original seed. Container validation remains pending.

The bundled issue judge was exercised with Qwen3-4B in bfloat16 on an H100 using Transformers 5.15.1. For each task, a correct diagnosis passed with score 1.0, while an incorrect causal explanation and an incorrect healthy verdict each received score 0.0. The three inputs and detailed judgments are in `evaluations/issue/validation/`. Run `evaluations/issue/verifier/judge.py` with each input, the task oracle and judge prompt to repeat the procedure.
The recorded control takes about 6.6 minutes for training and held-out evaluation, excluding asset preparation. The verifier allows 30 minutes.
Preparation downloads approximately 2.6 MB of data and a 1 GB public model separately.
