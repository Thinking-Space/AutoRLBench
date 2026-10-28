# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between
`d45a68d588b46c1f308f21f93fce0708ad25b7b0` and
`6e3c39d610c9537a6e03726ec1dd58029b1e3313` (verl PR #7014), applied to the
workspace's verl checkout at `6f2aa24abfc01cc72811b376aeb3936503885096`.

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.get_per_tensor_param`, merged-LoRA branch
  - kind: `symbol`
  - observed: with `model.lora.merge=True` the state dict is captured inside
    `merged_lora_context(self.module, backup_adapters=True)`, but the tensors are
    only materialized later by the lazy `per_tensor_param` generator
    (`param.to(device).full_tensor()`), which the rollout weight sync consumes
    after `get_per_tensor_param` has returned. Under FSDP2, `state_dict()` returns
    DTensors that alias the live parameter storage, and on exit
    `merged_lora_context` restores the un-merged base weights in place
    (`base.weight.data.copy_(backups[...])`). Every sync therefore sends vLLM the
    base weights without the LoRA delta.
  - expected: every merged tensor is materialized (`full_tensor()` for DTensors,
    a clone for plain tensors) while the merge context is still open, for example
    by yielding from inside the `with merged_lora_context(...)` block, so the
    rollout receives `W_base + scaling * B @ A`.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`.
- The rollout policy never receives the trained LoRA delta, so held-out accuracy
  stays at the base model's level: 0.0857 at step 0, between 0.0758 and 0.0842 at
  every later validation, 0.0842 at step 60. The same recipe with the sync
  repaired reaches 0.7286 at step 5 and 0.7771 at step 60.
- Training reward (`critic/score/mean`) stays at the base model's strict-format
  rate: 0.126 at step 1, 0.128 at step 30, 0.102 at step 60.
- The actor itself keeps updating: `actor/pg_loss` is non-zero throughout,
  `actor/grad_norm` grows from 0.040 (step 1) to 0.35 (step 10), 4.3 (step 30)
  and 12.7 (step 60), and `actor/kl_loss` (actor vs. reference) reaches 7.8.
- On the same rollout tokens, after each sync and before the update, the rollout
  and the actor diverge: `rollout_corr/kl` (masked mean of
  `rollout_log_prob - old_log_prob` over response tokens) is 0.0004 at step 1,
  0.55 at step 10, 20.6 at step 30 and 81.5 at step 60;
  `rollout_corr/rollout_ppl` stays between 1.50 and 1.59 (the base model's) while
  `rollout_corr/training_ppl` goes from 1.59 to 19.8, 1.3e30 and inf.
  `training/rollout_probs_diff_mean` is a probability-space difference
  (`|exp(rollout_log_prob) - exp(old_log_prob)|`); it rises from 0.0034 to 0.66.
- Frozen evidence is in `outputs/train.log`, `outputs/console-history.jsonl`,
  `outputs/wandb-history.jsonl`, `outputs/rollouts/` and `outputs/val_generations/`.

## Trigger conditions

- `actor_rollout_ref.model.lora_rank > 0` with
  `actor_rollout_ref.model.lora.merge=True` (the adapter is merged into the
  weights pushed to vLLM instead of being sent as a LoRA request).
- FSDP2 actor (`actor.strategy=fsdp2`), whose `state_dict()` returns DTensors
  that alias parameter storage and whose `full_tensor()` gather is deferred into
  the generator.
- Every sync after the adapter has moved off its zero initialisation; the
  step-0 sync is unaffected because `B = 0`.

## Documentation reconciliation

- The growing actor KL, gradient norm and rollout/actor divergence are
  symptoms: the actor trains off-policy on samples from a rollout policy that
  is pinned to the base weights. Lowering the learning rate, raising the KL
  coefficient, adding importance-sampling correction, or changing LoRA
  rank/alpha does not let the rollout see the adapter.
- "Training did not improve" is the wrong attribution: the optimizer updates
  the adapter every step; the weight sync drops it.
- bf16 precision loss in the merge is not the cause. A precision gap would be
  small and constant from the first step, whereas here the rollout's perplexity
  on its own samples stays at the base model's value and the gap grows with
  training.
- The strict GSM8K `####` format is not the cause: the same reward with the
  sync repaired learns the format within 5 steps.
