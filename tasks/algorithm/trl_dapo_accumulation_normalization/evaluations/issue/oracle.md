# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

`src/trl/trainer/grpo_trainer.py`, `GRPOTrainer._compute_loss`: the DAPO branch divides each micro-batch numerator by the token count of the entire generation batch without scaling that denominator to the current gradient accumulation window. Here `steps_per_generation=4` and `gradient_accumulation_steps=2`, so the logged optimizer-step objective is about half the intended window-token objective. The missing training correction is equivalent to multiplying the denominator by `current_gradient_accumulation_steps / steps_per_generation`; evaluation correctly uses its unsplit batch denominator. The exact arithmetic mechanism, rather than a poor reward or held-out score alone, is the essential diagnosis.

## Observable impact

Every item in this section is derivable from the shipped workspace (`results/`). Equivalent observations are accepted; the exact values are corroborating details.

- The recorded run completes all 24 optimizer steps (`metrics_rank0.jsonl`, `train.log`).
- Captured objectives are zero on some optimizer steps; on the nonzero steps the recorded loss is about 0.5× the accumulation-window objective recomputed from `token_records/`, because the denominator is the full generation batch.
- Held-out evaluation reward is 0.1953125 (`final_metrics.json`), with truncation and zero-variance groups as secondary symptoms. The exact objective discrepancy is stronger evidence than the small held-out reward gap.

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
