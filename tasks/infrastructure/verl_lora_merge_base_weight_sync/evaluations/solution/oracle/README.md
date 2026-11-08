# Oracle — merged LoRA never reaches the rollout engine

## Defect

`verl/verl/workers/engine/fsdp/transformer_impl.py`, `get_per_tensor_param`, the
`lora.merge` branch (line ~852): with FSDP2 + LoRA merged sync, the code takes
`self.module.state_dict()` inside `merged_lora_context` and hands the dict to the
lazy per-tensor generator further down (`param.to(device).full_tensor()` only when
iterated). FSDP2's `state_dict()` returns **lazy DTensor references**, not
materialized tensors: they only turn into real weight copies when `full_tensor()`
runs, which happens *after* the generator is consumed — i.e. after
`merged_lora_context.__exit__` has already restored the un-merged base weights from
backup. The rollout engine (vLLM) therefore receives the **base weights at every
weight sync**, and the merged LoRA delta never reaches generation.

Observable signature in the frozen evidence (`workspace/outputs/`):

- `train.log`: `training/rollout_probs_diff_max: 1.0`,
  `rollout_actor_probs_pearson_corr ≈ 0.16-0.20` (rollout policy ≠ training policy),
  `rollout_corr/kl` growing to ~80 while `actor/ppo_kl` stays ~0 — the actor is
  being trained against rollouts from a different (base) policy.
- Held-out accuracy pinned at the base model's ~0.084 across all 60 steps
  (`val-core/openai/gsm8k/acc/mean@1`), while `critic/score/mean` on training
  batches drifts upward — the training updates are real but never served.
- Probe evidence (task creation): the defective sync delivers bit-identical base
  weights; a correct sync delivers `bf16(W_base + dW)` on 98.98% of elements.

## Fix (fix.diff)

Materialize every merged tensor **while the context is still open** and force a
real copy: replace the lazy dict of DTensor references with
`{name: param.to(get_device_id(), non_blocking=True).full_tensor().to(torch.bfloat16,
non_blocking=True).detach().clone() if isinstance(param, DTensor) else
param.detach().clone() for ...}` inside the `with merged_lora_context(...)`
block. The downstream generator then sees plain tensors and passes them
through unchanged.

Eager materialization alone is **not** sufficient. A probe run (2026-11-07,
1 GPU, lora_B perturbed with std=0.02 so merged ≠ base is unmistakable) showed
the engine-side dict reading merged values inside the context (q_proj abs-sum
78547.27) while the rollout worker received the **bit-exact base weight**
(78189.4375) at every sync cycle: on a single-rank mesh the chain
`param.to(device).full_tensor().to(torch.bfloat16)` returns a *view* of the
live module storage (`.to()` casts are no-ops on an on-device bf16 param), and
`merged_lora_context.__exit__` restores the base weights **in place**
(`param.data.copy_(backup)`), so the bucketed weight sender — which consumes
the tensors only after `get_per_tensor_param` has returned — reads base
values. `.detach().clone()` inside the context makes every delivered tensor a
real copy taken while the merged weights are still live. Upstream verl PR
#7014 guards the same aliasing class by holding the context open across the
sender's consumption (generator) and cloning plain tensors; the
clone-inside-the-context form here is the eager equivalent.

A second probe (same day) held both the raw chain result and the cloned copy
across the context exit: the raw tensor flipped to bit-exact base
(78189.4375) while the clone retained the merged value (78540.58), and the
rollout worker then received exactly the clone's merged abs-sum at every sync
cycle — the mechanism and the fix are both confirmed directly.

Disabling `lora.merge`, LoRA, or FSDP2 avoids the path instead of fixing it; the
rerun pins all three (`run_grpo_lora.sh` is byte-compared, and the fresh run's
config-dump lines `'strategy': 'fsdp2'`, `'merge': True`, `'lora_rank': 32`,
`'lora_alpha': 64` are required by the grader).

## Expected numbers

Defective (workspace as shipped): step-60 strict held-out accuracy ~0.084
(regrade of the frozen dump: 0.0887). Repaired: ~0.78. Gate: 0.4.

## Files

- `fix.diff` — the repair, applies with `patch -p1` from the workspace root.
- `code_target/run_grpo_lora.sh` — pristine launcher (byte-pin target).
- `recorded_comparison/RESULTS.md` — calibration results (filled after the
  2-arm calibration: defective + repaired).
