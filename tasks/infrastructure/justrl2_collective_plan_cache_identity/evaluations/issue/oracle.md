# Canonical issue oracle

Reconciled from the frozen evidence in `workspace/results/ppo_countdown/` and the codebase in `workspace/`.

Expected verdict: **unhealthy**.

## Root cause

- `miles/utils/collective_plan.py::_signature`
  - kind: `symbol`
  - observed: keys `_plans` by `(dist.get_backend(group), dist.get_world_size(group), dist.get_rank(group), tensor.device.type, tensor.device.index, tensor.dtype, _size_class(tensor.numel()))`, omitting the communicator `group` object itself. All three native CP/DP sum-reduction sites call `reduce_sum(tensor, group=...)` unconditionally: `miles/utils/ppo_utils.py::get_advantages_and_returns_batch` (`cp_group`), `miles/utils/distributed_utils.py::distributed_masked_whiten` (`stats_tensor` of 3 `float32` elements -> `"small"`, `dp_group`), and `miles/backends/training_utils/loss.py::compute_ess_ratio_contribution` (`partial_sums` shape `[4, 2]` = 8 `float32` elements -> `"small"`, `cp_group`). Under `context_parallel_size=2` and `dp_size=2` (`world_size=4`), during step-3 actor training after `distributed_masked_whiten` caches the `"small"` `dp_group` plan per rank, `compute_ess_ratio_contribution` executes for the first time: diagonal ranks (`rank 0` `(dp=0, cp=0)` and `rank 3` `(dp=1, cp=1)`) have equal intra-group ranks across DP and CP (`0 == 0` and `1 == 1`), so they hit the stale `dp_group` cache entry (`DATA_PARALLEL_GROUP`), whereas off-diagonal ranks (`rank 1` `(dp=0, cp=1)` and `rank 2` `(dp=1, cp=0)`) miss on differing intra-group rank (`1 != 0` and `0 != 1`) and call `cp_group` (`CONTEXT_PARALLEL_GROUP`). This creates a 4-rank cyclic NCCL deadlock (`0 -> 2 -> 3 -> 1 -> 0`) on `NumelIn=8` even though all call sites execute unconditionally with matching tensor shapes across ranks.
  - expected: include communicator `group` identity in `miles/utils/collective_plan.py::_signature(tensor, group)` so distinct `dp_group` and `cp_group` communicators never alias the same cached `partial(dist.all_reduce, group=...)` plan.

Blaming a conditional collective skip on empty CP shards, slow network/storage or raising `--distributed-timeout-minutes`, SGLang weight sync, critic warmup, Countdown reward parsing, claiming steps 0-2 never ran, or calling the run healthy earns no root-cause credit.

## Observable impact

Every item below is directly observable in `workspace/results/ppo_countdown/`.

- Initial evaluation and critic-only warmup pass cleanly before failing: `val_generations/0.jsonl` and `metrics.jsonl` show step-0 evaluation completes (`eval/countdown = 0.10352`, 1,024 held-out rows), and critic warmup updates complete cleanly at steps 0, 1, and 2 (`train/critic-value_loss = 0.122093 -> 0.148268 -> 0.091744`) plus `critic-step 3` (`0.077107`).
- Deterministic step-3 collective deadlock on first actor update: once step 3 exits `num_critic_only_steps=3` and enters `policy_loss_function -> compute_ess_ratio_contribution`, `train.log` records `Watchdog caught collective operation timeout: WorkNCCL(..., OpType=ALLREDUCE, NumelIn=8, NumelOut=8, Timeout(ms)=180000)` with `rank 0` in `DATA_PARALLEL_GROUP` (`SeqNum=2`) and `ranks 1, 2` in `CONTEXT_PARALLEL_GROUP` (`SeqNum=3`), terminating training via `ActorDiedError` with 0 actor updates (`train/pg_loss`) and no `val_generations/29.jsonl`.

## Trigger conditions

- 1-node 8-GPU colocated topology (`actor=4, critic=4`) with `--context-parallel-size 2` (`dp_size=2` x `cp_size=2`) and synchronous PPO (`num_critic_only_steps=3`, `--micro-batch-size 4`, `partial_sums.numel() = 8` in `"small"` class matching `stats_tensor.numel() = 3`).
- Both `distributed_masked_whiten` (`dp_group`, `"small"`) and `compute_ess_ratio_contribution` (`cp_group`, `"small"`) routing through `collective_plan.reduce_sum`, triggering on step 3 (`>= num_critic_only_steps`).

## Documentation reconciliation

- `metrics.jsonl` records `rollout/step` 0-3 and `train/critic-value_loss` for steps 0-3, proving individual `cp_group` and `dp_group` calls succeed during critic warmup; deadlock occurs strictly when actor `policy_loss_function` introduces the colliding `"small"` `cp_group` reduction alongside `dp_group`.
- `ActorDiedError` and the `180000 ms` `ProcessGroupNCCL` watchdog timeout in `train.log` reflect the cyclic communicator mismatch between `DATA_PARALLEL_GROUP` (`ranks 0,3`) and `CONTEXT_PARALLEL_GROUP` (`ranks 1,2`), not hardware failure or insufficient timeout.
