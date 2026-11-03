# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
  - kind: `symbol`
  - observed: after per-uid mean/std normalization, the length-`B` advantage
    vector is reshaped to `[1, B]`, padded or truncated to the response length,
    expanded across the batch, and multiplied by `response_mask`. The result is
    still `[B, S]`, but column `j` carries sample `j`'s advantage for every row,
    and positions past `B` are zero before the mask.
  - expected: `scores.unsqueeze(-1) * response_mask`, so row `i` is filled with
    sample `i`'s scalar advantage on its response tokens.

The policy gradient is therefore decoupled from each sample's own reward: a
correct response and an incorrect one in the same batch receive the same
per-position advantages, so the update does not push the policy toward the
rewarded responses.

## Trigger conditions

- `algorithm.adv_estimator=grpo` routes through `compute_grpo_outcome_advantage`
  (not the vectorized estimator). Train batch 16 x rollout `n=8` gives `B=128`;
  `data.max_response_length=4096`.
- The advantage tensor's shape still matches the policy loss, so training never
  fails with a broadcast error. The layout is inside the existing
  `torch.no_grad()` block; the defect is the advantage values, not a gradient
  leak.

## Observable impact

From the bundled run
(`workspace/logs/wandb/offline-run-20260926_105216-l7my3znq/files/output.log`,
with its config in `files/config.yaml` and `workspace/outputs/2026-09-26/`):

- Training reward is present: `critic/rewards/mean` averages 0.122 over steps
  1-284, no step is 0, and fifty-step means stay between 0.093 and 0.133.
  GRPO groups do contain reward variance.
- Held-out `val-core/taco/acc/mean@1` (184 rows) does not improve:
  0.0761 at step 0, 0.0489 at step 50, 0.0598 at step 100, 0.0707 at step 150,
  0.0435 at step 200, 0.0543 at step 250.
- Fingerprint of the misaligned advantages: on every row the masked positions
  `0..min(len, 128)` carry the same column-indexed advantages, so each row
  sums a prefix of the 128 group-normalized values, which together sum to
  about zero, instead of its own advantage times its length. The token-weighted
  `critic/advantages/mean` therefore stays near zero on every step (median
  absolute value 4.9e-4). Correctly aligned advantages give a token-weighted mean
  that tracks the length/reward correlation (median about 4e-3 to 9e-3 on
  comparable TACO runs).
- The run is otherwise stable: `actor/entropy` stays between 0.39 and 0.81,
  mean response length between 149 and 503 tokens, clip ratio about 0.

## Not the root cause

- The run stopping at step 284 of the configured 400 is a truncated job (about
  8 hours of logged step time, no exception). A truncated log is not the defect.
- `use_kl_loss=False`, `kl_loss_coef=0`, and `entropy_coeff=0` are the shared
  launch defaults for this TACO family. Entropy and length do not collapse in
  this run, so missing regularization does not explain it.
- `verl/utils/reward_score/taco.py`, the dataset, the model, the learning rate,
  and batch size are the healthy baseline. `[error] runtime: ...` lines in the
  log, including `IndexError('list index out of range')`, are exceptions inside
  generated programs that the scorer catches and records as failures.
- A `keepdim` / `unsqueeze(0)` that would make `[B]` broadcast against `[B, S]`
  is not the defect; that form would raise under `B=128` and `S=4096`.
- `prime_code` and agent-loop tool parsers are unused by this TACO launch.
