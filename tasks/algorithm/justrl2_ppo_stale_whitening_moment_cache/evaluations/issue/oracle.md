# Canonical issue oracle

Reconciled from the frozen evidence in `workspace/results/ppo_countdown/` and the codebase in `workspace/`.

Expected verdict: **unhealthy**.

## Root cause

- `miles/utils/moments.py::MomentReducer`
  - kind: `symbol`
  - observed: caches a mutable two-element zero buffer `self._origins[(values.device, torch.float32)]` and reduces chunk sums via `functools.reduce(add, blocks, origin)`. Inside `miles/utils/distributed_utils.py::distributed_masked_whiten` (invoked per 4-sample prompt group in `miles/backends/training_utils/loss.py::compute_advantages_and_returns`), `dist.all_reduce(moment_tensor, group=process_group)` mutates `moment_tensor` in place, while `count_tensor` is reduced separately. Under ZigZag `context_parallel_size=2` on short Countdown responses (`response_len <= prompt_len / 3`), whenever a rank's local shard for a 4-sample prompt group is empty (`values.numel() == 0`), `blocks` yields zero items so `functools.reduce(add, blocks, origin)` returns the cached `origin` tensor object itself (`moment_tensor is origin`). `dist.all_reduce(moment_tensor, group=dp_group)` then permanently overwrites `self._origins[(device, float32)]` on that rank with the group's reduced `[sum, sum_sq]`. Every subsequent empty local shard starts from and compounds that stale non-zero moment buffer while contributing zero to `count_tensor`, corrupting whitened advantages without crashing, deadlocking, or producing NaNs.
  - expected: allocate a fresh zero `origin = torch.zeros(2, device=values.device, dtype=torch.float32)` inside `MomentReducer.__call__` (never caching a mutable buffer across calls) so empty local shards contribute `[0, 0]` to `dist.all_reduce` without retaining cross-call state.

Blaming early step-9 accuracy (`eval@9 = 0.29102`), claiming the run is healthy because all 30 steps exit cleanly with `ess_ratio = 1.0` and `ppo_kl = 0.0`, blaming `dp_group` vs `cp_group` collective desync (all collectives execute unconditionally across identical process groups), blaming critic value-head training (`critic-value_loss = 0.0187-0.1417`), learning rate `lr=2e-6`, `gae_lambda_k=0.5`, SGLang weight sync, or Countdown reward parsing earns no root-cause credit.

## Observable impact

Every item below is directly observable in `workspace/results/ppo_countdown/`.

- Complete run with mid-horizon regression on `eval/countdown`: `metrics.jsonl` and `val_generations/{0,9,19,29}.jsonl` cover all 30 rollout steps (`0..29`, `actor_steps 3..29`). Held-out accuracy initially climbs from `0.10059` (step 0) to `0.29102` (step 9), then degrades monotonically to `0.26660` (step 19) and `0.21289` (`218/1024`) at step 29.
- Response-length bloat and rising truncation: `eval/countdown/truncated_ratio` rises monotonically across evaluations (`0.20703` at step 0 -> `0.62500` at step 9 -> `0.68652` at step 19 -> `0.72754` at step 29), while mean eval response length grows from `133.2` (step 0) to `416.9` tokens (step 29).
- Optimization divergence masked by normal on-policy metrics: after clean critic warmup (`steps 0..2`) and normal step 3 (`train/pg_loss = 0.0`, `train/grad_norm = 2.82`), starting at step 4 `train/pg_loss` blows up to `-623.6` (step 4), `-2947.5` (step 15), `-3070.99` (step 24 min), and `-1053.3` / `-345.7` (steps 28-29), and `train/grad_norm` spikes to `3524.0` (step 4), `9669.2` (step 14 peak), and `2525.5` (step 29), while `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, `train/train_rollout_kl = 0.001113` (step 29), and `train/critic-value_loss = 0.0187-0.1417` stay nominal with no NaNs or crashes in `train.log`.

## Trigger conditions

- ZigZag context parallelism (`--context-parallel-size 2`) with short Countdown responses (`response_len <= prompt_len / 3`), leaving CP rank 1's local shard empty (`values.numel() == 0`) for frequent 4-sample prompt groups under per-group `distributed_masked_whiten`.
- `MomentReducer.__call__` returning the cached mutable `_origins` tensor when `blocks` is empty, paired with in-place `dist.all_reduce(moment_tensor, group=process_group)`.

## Documentation reconciliation

- Non-zero `eval@9` accuracy (`0.29102`) and clean completion reflect real early progress before accumulated stale-moment bias degrades the policy; claiming training failed from step 0 or crashed is false.
- Clean `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, and `train/critic-value_loss` reflect intact weight sync, ratio bounds, and return fitting — none detect in-place mutation of a module-level moment cache.
