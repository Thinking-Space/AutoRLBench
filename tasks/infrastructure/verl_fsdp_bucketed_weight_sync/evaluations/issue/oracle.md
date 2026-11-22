# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

*Staging note: both defects below are staged for this task. Upstream verl at commit `4b9c14f2` is correct at both sites — `build_optimizer` passes `parameters` straight to the optimizer class, and `on_bucket_received(weights)` is invoked inside the receive loop; `evaluations/solution/oracle/fix.diff` restores those upstream forms.*

- `verl/workers/config/optimizer.py::build_optimizer`
  - kind: `symbol`
  - observed: groups parameters into `param_groups = {}` keyed by `(param.device, param.dtype, param.ndim)`, overwriting the dict entry at each collision instead of appending (`param_groups[key] = {"params": [param]}`). Under FSDP2 (`strategy="fsdp2"`), trainable DTensor parameters share uniform device, `fp32` dtype, and dimensionality (2D projection matrices vs 1D RMSNorm vectors), collapsing 338 trainable parameters down to only 2 survivors passed to `AdamW` (`model.layers.27.mlp.down_proj.weight` and `model.norm.weight`), freezing 99.4% of model weights while still emitting non-zero `actor/grad_norm` and optimizer step updates without raising an error.
  - expected: pass all trainable parameters to `optimizer_cls(parameters, **optimizer_args)` without overwriting entries on shared metadata keys.

- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py::BucketedWeightReceiver.receive_weights`
  - kind: `symbol`
  - observed: places `on_bucket_received(weights)` and `del weights, tensor` outside the `while True:` bucket loop after the `if metadata["is_last"]: break` exit, while reinitializing `weights, tensor = [], None` on every bucket iteration. When `Qwen2.5-1.5B-Instruct` weights (`3,389.5 MiB` bf16) span two `2,048 MB` ZMQ transfer buckets (`update_weights_bucket_megabytes=2048`), the first bucket (`2,025.5 MiB`, 214 tensors from `embed_tokens` through `layers.17`) is acknowledged on ZMQ without applying it to the vLLM worker, so only the trailing `is_last` bucket (`1,364.0 MiB`, 125 tensors in `layers.17..27`, `norm`, `lm_head`) reaches vLLM while `update_weights` reports normal completion (`update_weights done`).
  - expected: invoke `on_bucket_received(weights)` inside the `while True:` loop before `get_torch_device().synchronize()` and `self.socket.send(b"")` (and release `del weights, tensor` per bucket) so every transferred bucket updates the rollout model.

Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.

Blaming `lr=2e-6`, `kl_loss_coef=0.001`, `gpu_memory_utilization=0.35`, async vLLM mode, Countdown AST reward parsing, W&B teardown `ConnectionResetError`, claiming `"training did not improve"`, or calling the run healthy earns no root-cause credit.

## Observable impact

Every item below is derivable from the shipped workspace. Equivalent observations are accepted; exact values and all bullets are corroboration, not separate hard requirements.

- Primary source evaluation metric: held-out `val-core/Jiayi-Pan/Countdown-Tasks-3to4/acc/mean@1` moves only slightly across 45 steps (`0.0117` at step 0 → `0.0146` at step 15 → `0.0205` at step 30 → `0.0410` at step 45, matching `val_generations` 42/1024) because only the 2 trailing tensors surviving `build_optimizer` sit in the second `is_last` ZMQ bucket and sync to vLLM.
- Suppressed policy divergence and stalled reward/entropy: `actor/kl_loss` barely moves (`8.5e-6` at step 1 → `0.00207` at step 45), `actor/ppo_kl` stays near zero (`-1.05e-6` → `-5.22e-6`) because actor and rollout share identical weights on both frozen and surviving parameters, `actor/pg_clipfrac` stays `0.0` throughout training, `actor/entropy` stays high (`0.688` → `0.673`), `critic/score/mean` stalls (`0.0391` → `0.0371`), and `actor/grad_norm` looks normal (`0.378` → `1.072`).

## Trigger conditions

- `actor_rollout_ref.actor.strategy=fsdp2` (`FSDP2`) with uniform DTensor `(device, dtype, ndim)` attributes across transformer layers inside `build_optimizer`.
- `actor_rollout_ref.rollout.name=vllm` colocated async bucketed weight sync with `Qwen2.5-1.5B-Instruct` bf16 weights (`3,389.5 MiB`) exceeding `update_weights_bucket_megabytes=2048` (`> 1` ZMQ transfer bucket).

## Documentation reconciliation

- `update_weights done` in `train.log` only reflects that the ZMQ loop drained and acknowledged all buckets via `self.socket.send(b"")`, masking that `bucket[0]` (`embed_tokens` through `layers.17`) was overwritten before `on_bucket_received(weights)`.
- Similarly, `actor/ppo_kl ≈ 0` and non-zero `actor/grad_norm` in the dual-defect run hide the broken ZMQ sync because the only two updated parameters (`layers.27.mlp.down_proj.weight` and `model.norm.weight`) both land in the trailing `is_last` bucket.
- The W&B offline `ConnectionResetError` at step 45 teardown occurs after training and final validation finish and is not a root cause.
