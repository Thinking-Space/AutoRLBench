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

Every item below is directly observable in `workspace/results/ppo_countdown/`. These three
bullets are the key evidence items: exact values are illustrative, approximate figures or
the correct direction suffice, and the per-step value ladders are illustrative detail rather
than additional scored items.

- Complete run with a mid-horizon regression on `eval/countdown`: all 30 rollout steps and
  every evaluation complete cleanly; held-out accuracy climbs from ~`0.10` (step 0) to a
  mid-run peak (~`0.29` around step 9), then degrades to ~`0.21` (`218/1024`) at step 29.
- Response-length bloat and rising truncation: `truncated_ratio` rises across evaluations
  (from ~`0.21` at step 0 to ~`0.73` at step 29) as mean response length grows several-fold.
- Optimization divergence masked by normal on-policy metrics: after clean critic warmup and
  a normal first actor step, `train/pg_loss` blows up to large negative values (down to
  about `-3070`) and `train/grad_norm` spikes into the thousands from step 4 onward, while
  `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, and `train/critic-value_loss` stay nominal,
  with no NaNs or crashes anywhere in `train.log`.

## Trigger conditions

- ZigZag context parallelism (`--context-parallel-size 2`) with Countdown responses short
  relative to their prompts, leaving CP rank 1's local shard empty (`values.numel() == 0`)
  for frequent 4-sample prompt groups under per-group `distributed_masked_whiten`.
- `MomentReducer.__call__` returning the cached mutable `_origins` tensor when `blocks` is empty, paired with in-place `dist.all_reduce(moment_tensor, group=process_group)`.

## Documentation reconciliation

- Non-zero mid-run accuracy and clean completion reflect real early progress before accumulated stale-moment bias degrades the policy; claiming training failed from step 0 or crashed is false.
- Clean `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, and `train/critic-value_loss` reflect intact weight sync, ratio bounds, and return fitting — none detect in-place mutation of a module-level moment cache.
