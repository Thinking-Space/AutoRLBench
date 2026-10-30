# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._compute_loss`: the DAPO denominator uses the token count of the entire generation batch without scaling it to the current gradient accumulation window. With steps_per_generation=4 and gradient_accumulation_steps=2, the loss and raw gradient for identical inputs are half the intended accumulation-window objective. During training multiply the denominator by current_gradient_accumulation_steps / steps_per_generation. Evaluation uses its unsplit batch denominator.

## Observable impact

Both arms complete 24 optimizer steps. On nonzero captured objectives, the defective loss/reference ratio is approximately 0.5 and the control ratio is approximately 1.0. The independent objective comparison passes 12/24 optimizer steps in the defective run and 24/24 in the control. The held-out rewards are 0.1953125 and 0.20703125 respectively. This single paired run does not establish a general accuracy effect.

## Trigger conditions

The effective run uses generation batches spanning four microbatches and accumulation windows spanning two.
The distributed case has two GPU processes and a reward metric emitted only on rank 0.

## Documentation reconciliation

Reject diagnoses of no learning, corrupted GSM8K labels, an incorrect numerical-answer reward, insufficient model size, or a guaranteed twofold Adam parameter update. Adam adaptation and gradient clipping can compensate for loss scaling. The strongest evidence is the exact objective discrepancy, not the small held-out reward gap.

## Source

The reproduction starts from TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144 and removes only the training denominator correction corresponding to upstream PR #6024. This is an isolated reintroduction of a documented historical defect, rather than a claim that the current upstream release remains affected.
https://github.com/huggingface/trl/pull/6024
