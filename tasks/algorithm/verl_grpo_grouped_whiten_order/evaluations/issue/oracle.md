# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `verl/utils/torch_functional.py::grouped_whiten` (and `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`)
  - kind: `symbol`
  - observed: `compute_grpo_outcome_advantage` converts prompt UIDs via `as_torch_index` and calls
    `verl_F.grouped_whiten`, which sorts `values` by `order = torch.argsort(group_ids, stable=True)`,
    standardizes within each prompt group, and returns `torch.gather(whitened, 0, order)`. Because
    `torch.gather` indexes by the forward permutation `order` instead of inverting it (e.g.
    `torch.empty_like(whitened).scatter_(0, order, whitened)` or
    `whitened.index_select(0, torch.argsort(order))`), standardized advantages are assigned to the
    wrong trajectories whenever repeated prompt samples are non-contiguous, while batch advantage
    mean/variance remain plausible.
  - expected: scatter standardized advantages back using the inverse permutation via an out-of-place
    buffer (or compute per-UID standardization directly in `compute_grpo_outcome_advantage`).

- `verl/trainer/ppo/core_algos.py::kl_penalty`
  - kind: `symbol`
  - observed: replaces `if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):` with
    `if kl_penalty in ("kl", "k1", "abs", "mse", "k2"):`, routing the configured
    `kl_loss_type="low_var_kl"` into the straight-through backward branch below. In that branch,
    `seq_log_ratio = (logprob - ref_logprob).sum(dim=-1, keepdim=True)` and
    `backward_score = 0.5 * seq_log_ratio.square().expand_as(logprob)` are passed to
    `backward_score - backward_score.detach() + forward_score.detach()`. Forward `actor/kl_loss`
    still equals per-token `k3` (`forward_score`), while squaring the sequence-summed log-ratio and
    broadcasting across tokens scales backward KL gradients by ~`L_i` per token (~`L_i^2` per
    sequence), pulling updates strongly back toward the reference model.
  - expected: return `forward_score` when `not kl_penalty.endswith("+")` (so `low_var_kl` does
    not enter the straight-through branch) and keep per-token
    `backward_score = 0.5 * (logprob - ref_logprob).square()` for `+`-suffixed variants.

Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.

Blaming `kl_loss_coef=0.001`, `lr=2.5e-6`, W&B shutdown `ConnectionResetError`, data filtering,
reward extraction, or calling the run healthy earns no root-cause credit.

## Observable impact

Every item below is derivable from the shipped workspace. Equivalent observations are accepted; exact values and all bullets are corroboration, not separate hard requirements.

- Primary source evaluation metric: logged `val-core/openai/gsm8k/acc/mean@1` fails to gain over
  50 steps (`0.4359` at step 0 → `0.4117` at step 25 → `0.4337` at step 50; `val_generations`
  numeric accuracy `0.4625` → `0.4579` → `0.4610`).
- Suppressed forward KL and stalled reward/entropy: `actor/kl_loss` stays mild (`0.00029` at
  step 1 → `0.00795` at step 50), `actor/pg_clipfrac` stays small (`0.00060` → `0.00082`),
  `actor/grad_norm` stays normal (`0.805` → `1.280`), `actor/entropy` remains high (`0.485` →
  `0.456`), and training reward (`critic/score/mean`) fails to climb (`0.4102` → `0.3594`).
- Batch advantage summary statistics remain plausible (`critic/advantages/mean = -0.0204` at step 1
  and `-0.0296` at step 50) despite row-level misassignment in `grouped_whiten`.

## Trigger conditions

- GRPO with `n=8` rollouts per prompt and sequence-length batch balancing, so prompt UIDs are
  non-contiguous before `grouped_whiten` (`order` is not an involution).
- `actor_rollout_ref.actor.use_kl_loss=True` with `kl_loss_type="low_var_kl"`, which enters the
  sequence-squared straight-through backward branch in `kl_penalty`.

## Documentation reconciliation

- `actor/kl_loss` (`0.00795`) and `actor/loss == actor/pg_loss + 0.001 * actor/kl_loss` reflect
  `forward_score` (`k3`) from the detached branch and hide the sequence-summed backward KL gradient.
- The W&B offline `ConnectionResetError` at step 50 teardown occurs after training and final
  validation finish and is not a root cause.
