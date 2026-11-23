# Canonical issue oracle

Reconciled from the frozen evidence in `workspace/results/ppo_countdown/` and the diff between `task2_buggy` and `task2_fixed` (`evaluations/solution/oracle/fix.diff`).

Expected verdict: **unhealthy**.

## Root cause

*Staging note: this defect is staged for this task against pinned `JustRL-II @ 75814318`, not an unpatched upstream bug. Upstream `JustRL-II` computes local moments directly inside `miles/utils/distributed_utils.py::distributed_masked_whiten`; `evaluations/solution/oracle/fix.diff` removes the cached `self._origins` dict from `miles/utils/moments.py::MomentReducer` and allocates a fresh `torch.zeros(2, device=values.device, dtype=torch.float32)` buffer per call. The workspace also carries image-compatibility updates in `arguments.py`, `actor.py` (`process_group_compat.py`), `sglang_engine.py`, `update_weight_from_distributed/mixin.py`, `update_weight_from_tensor.py`, `loss.py` (`logdiff` empty-shard guard), `tracking_utils/__init__.py`, and `rollout.py`, plus per-4-sample group DP whitening in `loss.py` (`compute_advantages_and_returns`); none of those cause stale cross-call accumulation when `MomentReducer` uses a fresh zero origin.*

- `miles/utils/moments.py::MomentReducer`
  - kind: `symbol`
  - observed: caches a mutable two-element zero buffer `self._origins[(values.device, torch.float32)]` and reduces chunk sums via `functools.reduce(add, blocks, origin)`. Inside `miles/utils/distributed_utils.py::distributed_masked_whiten` (invoked per 4-sample prompt group in `compute_advantages_and_returns`), `dist.all_reduce(moment_tensor, group=process_group)` mutates `moment_tensor` in place, while `count_tensor` is reduced separately. Under ZigZag `context_parallel_size=2` on short Countdown responses (`response_len <= prompt_len / 3`, ~78.5% of step-0 eval responses on CP rank 1), whenever a rank's local shard for a 4-sample prompt group is empty (`values.numel() == 0`), `blocks` yields zero items so `functools.reduce(add, blocks, origin)` returns the cached `origin` tensor object itself (`moment_tensor is origin`). `dist.all_reduce(moment_tensor, group=dp_group)` then permanently overwrites `self._origins[(device, float32)]` on that rank with the group's reduced `[sum, sum_sq]`. Every subsequent empty local shard starts from and compounds that stale non-zero moment buffer while contributing zero to `count_tensor`, corrupting whitened advantages without crashing, deadlocking, or producing NaNs (`train/pg_loss` diverges to negative thousands and `train/grad_norm` spikes into the thousands while `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, and critic loss stay normal).
  - expected: allocate a fresh zero `origin = torch.zeros(2, device=values.device, dtype=torch.float32)` inside `MomentReducer.__call__` (never caching a mutable buffer across calls) so empty local shards contribute `[0, 0]` to `dist.all_reduce` without retaining cross-call state.

Blaming early step-9 accuracy (`eval@9`), claiming the run is healthy because all 30 steps exit `0` with `ess_ratio = 1.0` and `ppo_kl = 0.0`, blaming `dp_group` vs `cp_group` collective desync (all collectives execute unconditionally across identical process groups), blaming critic value-head training (`critic-value_loss = 0.019-0.133`), learning rate `lr=2e-6`, `gae_lambda_k=0.5`, SGLang weight sync, or Countdown reward parsing earns no root-cause credit.

## Observable impact

Every item below is derivable from the shipped workspace. Equivalent observations are accepted; exact values and all bullets are corroboration, not separate hard requirements.

- Primary source evaluation metric (`eval/countdown`): completes all 30 steps (`exit_code = 0`) and even improves early at step 9 (`0.10059` at step 0 -> `0.29102` at step 9, exceeding the clean seed-42 control's `0.05664`), before silently sliding downhill to `0.26660` at step 19 and collapsing to **`0.21289`** at step 29 (versus `0.41699` in control seed 42; seed 43 shows the same pattern: `0.09766 -> 0.23242 -> 0.13086 -> 0.03418` vs `0.34473` in control).
- Length bloat and truncation surge: mean eval response length stays elevated (`398.5` at step 9 -> `416.9` at step 29) and `eval/countdown/truncated_ratio` rises from `0.20703` to `0.72754` (control seed 42: `0.52051`).
- Optimization divergence masked by normal on-policy indicators: after clean critic warmup (`steps 0..2`) and normal step 3 (`train/pg_loss = 0.0`, `train/grad_norm = 2.816`), once step-4 updates run on stale cached advantages `train/pg_loss` drifts to `-623.6` (step 4), `-2947.5` (step 15), `-3070.99` (step 24 min), and `-1053.3` / `-345.7` (steps 28-29), and `train/grad_norm` peaks at `9669.2` (step 14) / `2525.5` (step 29; control: `1.44-4.42`), while `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, `train/pg_clipfrac = 0.0`, `train/train_rollout_kl ≈ 0.0011`, and `train/critic-value_loss = 0.0187-0.1417` give zero error or staleness warning.

## Trigger conditions

- ZigZag context parallelism (`--context-parallel-size 2`) with prompt-length-dominated short Countdown responses (`response_len <= prompt_len / 3`), leaving CP rank 1's local shard empty (`values.numel() == 0`) for frequent 4-sample prompt groups under per-group `distributed_masked_whiten`.
- `MomentReducer.__call__` returning the cached mutable `_origins` tensor when `blocks` is empty, paired with in-place `dist.all_reduce(moment_tensor, group=process_group)`.

## Documentation reconciliation

- Non-zero `eval@9` accuracy (`0.29102`) and clean `exit_code = 0` are genuine partial progress before stale advantage bias compounds across steps; "training did not learn at all" is false.
- Healthy `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, and `train/critic-value_loss` reflect intact weight sync, unchanged log-prob consistency, and clean return regression on CP/critic respectively — none of them check for in-place mutation of a module-level moment cache.
