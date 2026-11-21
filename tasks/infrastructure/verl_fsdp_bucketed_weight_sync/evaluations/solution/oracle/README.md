# Canonical solution oracle

## Root cause

- `verl/workers/config/optimizer.py::build_optimizer`
  - kind: `symbol`
  - observed: `param_groups[key] = {"params": [param]}` keyed by `(param.device, param.dtype, param.ndim)` overwrites colliding entries, leaving only 2 of 338 FSDP2 DTensor parameters (`model.layers.27.mlp.down_proj.weight` and `model.norm.weight`) in AdamW.
  - expected: pass `parameters` directly to `optimizer_cls(parameters, **optimizer_args)`.

- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py::BucketedWeightReceiver.receive_weights`
  - kind: `symbol`
  - observed: calls `on_bucket_received(weights)` and `del weights, tensor` only after exiting the `while True:` ZMQ bucket loop on `metadata["is_last"]`, discarding every earlier bucket (`bucket[0]`, `2,025.5 MiB` / 214 tensors from `embed_tokens` through `layers.17`).
  - expected: invoke `on_bucket_received(weights)` inside the loop before `get_torch_device().synchronize(); self.socket.send(b"")` and free `del weights, tensor` on each iteration.

## Observable impact

- Primary source evaluation metric: `heldout/countdown_exact_acc@45` stalls at `0.0410` / `0.0449` (`42/1024` and `46/1024`; `actor/kl_loss`: `0.00207` / `0.00245`, `actor/ppo_kl`: `-5e-6` / `3e-5`) versus `0.4658` / `0.4883` (`477/1024` and `500/1024`) in the control.

## Trigger conditions

- Same as `evaluations/issue/oracle.md`.

## Required remediation

- `verl/workers/config/optimizer.py` & `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`: return `optimizer_cls(parameters, **optimizer_args)` in `build_optimizer`, and move `on_bucket_received(weights)` plus `del weights, tensor` inside the `while True:` loop of `BucketedWeightReceiver.receive_weights` (`fix.diff`).

Machine grading checks only the fresh rerun logs. It re-scores `experiment_logs/val_generations/45.jsonl`
(Countdown AST check, `>= 1,000` rows, pass at `>= 0.38`), verifies continuous `metrics.jsonl` rows
across steps `0..45` with `val-core/Jiayi-Pan/Countdown-Tasks-3to4/acc/mean@1` at step 45, checks
`run_grpo_countdown.sh`, pinned files, and generated parquets. Fixing only `receive_weights` ends at
`0.0459` / `0.0664`; fixing only `build_optimizer` triggers stale-rollout KL lag (`actor/ppo_kl`
`0.0159` / `0.0764`, `actor/kl_loss` `0.7357` / `1.9251`) and stalls at `0.1650` / `0.2695 < 0.38`.
