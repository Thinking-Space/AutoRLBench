# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._compute_loss`: the DAPO denominator uses the token count of the entire generation batch without scaling it to the current gradient accumulation window. With steps_per_generation=4 and gradient_accumulation_steps=2, the loss and raw gradient for identical inputs are half the intended accumulation-window objective. During training multiply the denominator by current_gradient_accumulation_steps / steps_per_generation. Evaluation uses its unsplit batch denominator.

## Observable impact

Every item in this section is derivable from the shipped workspace
(`results/`). The counterfactual control run is author-side reference and
lives in Documentation reconciliation below.

- The recorded run completes all 24 optimizer steps (`metrics_rank0.jsonl`,
  train.log).
- Captured objectives are zero on 12 of the 24 optimizer steps; on the 12
  nonzero steps the recorded loss is ≈ 0.5× the accumulation-window objective
  recomputed from the recorded per-token data (`token_records/`).
- Held-out evaluation reward of the recorded run: 0.1953125
  (`final_metrics.json`).

This single paired run does not establish a general accuracy effect.

## Trigger conditions

The recorded run uses one GPU, generation batches spanning four microbatches and accumulation windows spanning two.

## Documentation reconciliation

- Control reference (author-side; the same reproduction with the training
  denominator correction restored — not shipped in the workspace, and
  comparisons to it are not evidence items): the control completes all 24
  optimizer steps with the loss/reference ratio at approximately 1.0 on every
  nonzero step, the independent objective comparison passes 24/24 steps, and
  its held-out evaluation reward is 0.20703125.

Reject diagnoses of no learning, corrupted GSM8K labels, an incorrect numerical-answer reward, insufficient model size, or a guaranteed twofold Adam parameter update. Adam adaptation and gradient clipping can compensate for loss scaling. The strongest evidence is the exact objective discrepancy, not the small held-out reward gap.

## Source

The reproduction starts from TRL commit 03b22f5f93868a8aa73c5089ab99fbceb05ae144 and removes only the training denominator correction corresponding to upstream PR #6024. This is an isolated reintroduction of a documented historical defect, rather than a claim that the current upstream release remains affected.
https://github.com/huggingface/trl/pull/6024
