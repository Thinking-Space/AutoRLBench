# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only
`verl/workers/engine/fsdp/transformer_impl.py` differs).

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.forward_backward_batch`
  - kind: `symbol`
  - observed: `self.module.zero_grad(set_to_none=True)` runs before every micro-batch's backward,
    so the gradients accumulated from the earlier micro-batches of a mini-batch are discarded and
    `optimizer_step` applies only the last micro-batch's gradient. With
    `ppo_mini_batch_size=32` prompts × `rollout.n=8` = 256 samples and
    `ppo_micro_batch_size_per_gpu=32`, every optimizer step uses 32 of 256 samples (4 prompt
    groups); 7/8 of each batch's rollouts never reach the optimizer.
  - expected: gradients accumulate across all micro-batches of the mini-batch (one
    `optimizer_zero_grad` per `train_batch`, as in the control).

Naming the per-micro-batch gradient reset with its causal role (only the last micro-batch's
gradient survives → each update is estimated from 4 prompt groups instead of 32 → AdamW still
takes full-size steps, so the policy drifts at the normal rate but along a much noisier direction →
held-out accuracy does not improve and falls back to its starting level) is a complete
attribution. Do not require additional causes.

**The gradient reset inside the micro-batch loop is required.** "Learning is noisy", "the
gradient norm is small", "the learning rate is wrong" or "entropy collapse" without locating the
reset earns at most partial root-cause credit. Blaming the format loophole (the prompt asks for
`####`, the flexible scorer takes the last number and the policy drifts to `\boxed{}`; the control
has the same loophole and reaches 0.5580), the reward function, the data filter, the KL settings,
or calling the run healthy is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: logged `val-core/openai/gsm8k/acc/mean@1` 0.434 (0) → 0.453
  (10) → 0.462 (20) → 0.450 (30) → 0.457 (40) → 0.454 (50) → 0.451 (60). Re-scoring
  `val_generations/<step>.jsonl` gives 0.4610, 0.4678, 0.4723, 0.4541, 0.4579, 0.4549, 0.4511: a
  peak at step 20 and a final value below step 0.
- Training reward (`critic/score/mean`) rises 0.389 (step 1) → 0.504 (20) → 0.602 (60) while
  held-out accuracy does not follow.
- The policy moves at a normal rate: `actor/entropy` 0.49 → 0.10 and `actor/kl_loss` 0.083 at step
  60 (`kl_loss_coef=0.001`), so the run does not look stuck.
- `actor/grad_norm` stays at 0.12-0.36 for all 60 steps: it is the norm of the one surviving
  micro-batch gradient, which the loss normalization scales by that micro-batch's share of the
  mini-batch's tokens (about 1/8).

## Trigger conditions

- Gradient accumulation (`ppo_micro_batch_size_per_gpu` smaller than the per-GPU mini-batch); with
  a single micro-batch per mini-batch the reset is harmless.
- An adaptive optimizer (AdamW) normalizes the update size, so the lost data shows up as a noisy
  update direction rather than as a stalled policy.

## Documentation reconciliation

- The code comment justifies the reset as a peak-memory measure; it discards the accumulated
  gradient instead. The flat held-out accuracy, the small gradient norm and the train/held-out
  divergence are symptoms of the same defect.
