# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only one line of
`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` differs).

Expected verdict: **unhealthy**.

## Root cause

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
  - kind: `symbol`
  - observed: `scores[i] = (scores[i] - id2mean[index[i]]) / id2std[index[i]]` — the group std
    without `epsilon`. Groups whose eight rewards are identical (18-21 of 64 per sampled step) have
    std 0 and get NaN advantages; the token-mean policy loss of every mini-batch is NaN, the
    gradient norm is NaN, and `workers/engine/fsdp/transformer_impl.py` zeroes the gradients and
    skips the optimizer step (`WARN: grad_norm is not finite: nan`, 120 times = 2 mini-batches × 60
    steps). The policy never changes.
  - expected: `/ (id2std[index[i]] + epsilon)` (upstream), which gives zero-variance groups a zero
    advantage.

Naming the missing epsilon with its causal role (std 0 → NaN advantages → NaN loss and gradient →
every update skipped → no learning) is a complete attribution. Do not require additional causes.

**The division without epsilon is required.** "The loss is NaN", "updates are skipped" or "the
policy does not learn" without locating the division earns at most partial root-cause credit.
Blaming bf16/FSDP numerics, the learning rate, gradient explosion, the KL coefficient, the reward
function, the data, or calling the run healthy is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` stays at 0.422-0.443 over
  steps 0-60; re-scored from `val_generations` (last number vs `gts`): 0.4617 (0), 0.4723 (10),
  0.4678 (20), 0.4587 (30), 0.4526 (40), 0.4602 (50), 0.4564 (60).
- `actor/pg_loss`, `actor/grad_norm`, `critic/advantages/mean|max|min` are `nan` at every step
  (`null` in `metrics.jsonl`); `train.log` has 120 `WARN: grad_norm is not finite: nan` lines.
- `actor/kl_loss` is exactly 0 and `actor/ppo_kl`/`actor/pg_clipfrac` are 0 for all 60 steps; entropy
  0.44-0.54 and response length 294-327 tokens without trend; `critic/score/mean` 0.30-0.39 without
  trend.
- `rollouts/{1,30,60}.jsonl`: 18, 20 and 21 of 64 groups have eight identical scores.

## Trigger conditions

- GRPO with std normalization and any group whose rewards are all equal, which binary rewards make
  common.
- verl's FSDP engine silently skips non-finite steps instead of failing, so the run completes.

## Documentation reconciliation

- The NaN loss/gradient, the zero KL and the flat accuracy are symptoms of the same division; the
  skipped steps are verl's safeguard working as designed, not a second defect.
