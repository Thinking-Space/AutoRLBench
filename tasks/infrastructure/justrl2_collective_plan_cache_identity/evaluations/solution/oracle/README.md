# Canonical solution oracle

## Root cause

- `miles/utils/collective_plan.py::_signature`
  - kind: `symbol`
  - observed: keys `_plans` by `(dist.get_backend(group), dist.get_world_size(group), dist.get_rank(group), tensor.device.type, tensor.device.index, tensor.dtype, _size_class(tensor.numel()))`, omitting the communicator `group` object itself. On `context_parallel_size=2` × `data_parallel_size=2` (`world_size=4`), `distributed_masked_whiten` (`stats_tensor` of 3 `float32` elements -> `"small"`, `dp_group`) and `compute_ess_ratio_contribution` (`partial_sums` shape `[4, 2]` = 8 `float32` elements -> `"small"`, `cp_group`) execute consecutively starting at step 3 (`num_critic_only_steps=3`). Diagonal ranks (`rank 0` `(dp=0, cp=0)` and `rank 3` `(dp=1, cp=1)`) have equal intra-group ranks across `dp_group` and `cp_group` (`0 == 0`, `1 == 1`) and hit the cached `"small"` `dp_group` plan (`DATA_PARALLEL_GROUP`), while off-diagonal ranks (`rank 1` `(dp=0, cp=1)` and `rank 2` `(dp=1, cp=0)`) miss (`1 != 0`, `0 != 1`) and invoke `cp_group` (`CONTEXT_PARALLEL_GROUP`), creating a 4-rank cyclic NCCL deadlock (`0 -> 2 -> 3 -> 1 -> 0`) on `NumelIn=8`.
  - expected: include `group` in the tuple returned by `_signature(tensor, group)` so `dp_group` and `cp_group` never share cached reduction plans.

## Observable impact

- Defective runs complete step-0 eval (`0.1035` on seed 42, `0.1045` on seed 43) and critic warmup updates (`0.122093 -> 0.148268 -> 0.091744 -> 0.077107` on seed 42), then hang and abort in `compute_ess_ratio_contribution` on the first actor step (`step 3`, `WorkNCCL(SeqNum=2/3, OpType=ALLREDUCE, NumelIn=8, Timeout(ms)=180000)`, `exit_code = 1`, `0` actor updates).
- Control (`fix.diff`) completes all 30 steps (`exit_code = 0`) and reaches `heldout/countdown_exact_acc@29 = 0.4453` (`456/1024`, seed 42) and `0.4150` (`425/1024`, seed 43).

## Trigger conditions

- Same as `evaluations/issue/oracle.md`.

## Required remediation

- Include `group` in `miles/utils/collective_plan.py::_signature` (`evaluations/solution/oracle/fix.diff`).

Machine grading checks only the fresh rerun logs. It re-scores `experiment_logs/val_generations/29.jsonl`
(Countdown exact rational AST check, `step == 29`, `dataset == "countdown"`, zero malformed rows,
exact `1,024` unique held-out test rows matching `data/countdown/test.jsonl`, pass at `>= 0.32`),
verifies continuous `rollout/step` `0..29` with exact actor steps `3..29` (`train/pg_loss`), critic
steps `0..29`, and matching `eval/countdown@29` in `metrics.jsonl`, and enforces pinned
`run_justrl2_ppo.sh`, Countdown source/script hashes, and raw-to-JSONL split consistency.
