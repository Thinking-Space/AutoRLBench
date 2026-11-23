# Canonical solution oracle

## Root cause

- `miles/utils/moments.py::MomentReducer`
  - kind: `symbol`
  - observed: caches a mutable two-element zero tensor `self._origins[(values.device, torch.float32)]` and uses it as the `functools.reduce(add, blocks, origin)` initial value. Inside `distributed_masked_whiten` (invoked per 4-sample prompt group in `compute_advantages_and_returns`), whenever a rank's local ZigZag CP shard contains zero elements (`values.numel() == 0`, frequent on CP rank 1 for short Countdown responses), `reduce` returns the cached `origin` tensor object itself (`moment_tensor is origin`), which `dist.all_reduce(moment_tensor, group=dp_group)` mutates in place and permanently retains as non-zero state for subsequent empty-shard calls while contributing `0` to `count_tensor`.
  - expected: allocate a fresh `origin = torch.zeros(2, device=values.device, dtype=torch.float32)` per `MomentReducer.__call__` invocation so empty local shards never carry mutated state from prior reductions.

## Observable impact

- Defective runs exit `0` across all 30 steps without deadlocks or NaNs and even lead at step 9 (`0.2910` vs `0.0566` on seed 42; `0.2324` vs `0.2188` on seed 43) before collapsing to `heldout/countdown_exact_acc@29 = 0.2129` (`218/1024`, seed 42) and `0.0342` (`35/1024`, seed 43), alongside truncation surge (`0.7275` / `0.7656`), negative policy-loss blowup (`train/pg_loss` down to `-3070.99` / `-679.17`), and `train/grad_norm` spikes (`9669.23` / `3440.76` peak), while `train/ess_ratio = 1.0`, `train/ppo_kl = 0.0`, and `train/critic-value_loss` stay unremarkable.
- Control (`fix.diff`) reaches `heldout/countdown_exact_acc@29 = 0.4170` (`427/1024`, seed 42) and `0.3447` (`353/1024`, seed 43) with `train/pg_loss = 0.0000` and `train/grad_norm = 1.18–4.42`.

## Trigger conditions

- Same as `evaluations/issue/oracle.md`.

## Required remediation

- Allocate a fresh zero `origin` per call instead of caching `self._origins` in `miles/utils/moments.py::MomentReducer` (`evaluations/solution/oracle/fix.diff`).

Machine grading checks only the fresh rerun logs. It re-scores `experiment_logs/val_generations/29.jsonl`
(Countdown exact rational AST check, `>= 1,000` rows, pass at `>= 0.28`), verifies continuous
`metrics.jsonl` rows across rollout steps `0..29` with `>= 27` actor steps logging `train/pg_loss`
and an `eval/countdown` row at step `29`, and enforces pinned `run_justrl2_ppo.sh`, Countdown source/script
hashes, and raw-to-JSONL split consistency (`8,192` train / `1,024` disjoint test rows).
