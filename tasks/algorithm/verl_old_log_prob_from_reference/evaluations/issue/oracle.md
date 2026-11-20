# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed, budget and image; only `verl/trainer/ppo/ray_trainer.py`
differs, by one line).

Expected verdict: **unhealthy**.

## Root cause

- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer._compute_old_log_prob`
  - kind: `symbol`
  - observed: the function computes the batch's `old_log_probs` (and the `entropys` behind
    `actor/entropy`) with `self.ref_policy_wg.compute_ref_log_prob(batch_td)`, i.e. with the frozen
    **reference** model, instead of the actor (`self.actor_rollout_wg.compute_log_prob`). The PPO
    loss (`verl/workers/utils/losses.py::ppo_loss`, unchanged) therefore takes its importance ratio
    as π_θ / π_ref, and its clip (`clip_ratio=0.2`) becomes a per-token trust region around the base
    model: once a token's probability has moved more than 20% from the base model in the direction
    its advantage favours, it gets no policy gradient, while moves back toward the base model stay
    unclipped. The policy is held near the base model for the whole run and GRPO stops improving it
    after the first few steps.
  - expected: `old_log_probs` are recomputed by the actor (the policy that generated the rollouts)
    at the start of each step, as in the control; the reference model enters only through
    `_compute_ref_log_prob` and the KL loss term (`kl_loss_coef=0.001`).

Naming that the old/proximal log-probs come from the reference model rather than the actor, with
its causal role (PPO ratio and clip measured against the frozen base model → the clip pins the
policy near the base model → policy-gradient updates stall → held-out accuracy barely moves) is a
complete attribution. Do not require additional causes. Mentioning that the logged
`actor/entropy` is consequently the reference model's entropy is welcome but not required.

**The reference-model call in `_compute_old_log_prob` is required.** "Learning is too slow", "the
learning rate is too low", "the KL penalty is too strong", "the clip range is too tight" or "the
updates are over-clipped" without locating the reference model as the source of the old
log-probs earns at most partial root-cause credit. Blaming the DAPO overlong penalty (it subtracts a
penalty only for responses longer than 512 tokens; the control has the same penalty and improves),
the flexible last-number reward, the data, or calling the run healthy is wrong and gets no
root-cause credit. "Training did not improve at all" is also inaccurate: training score and
held-out accuracy rise slightly but stay far below what the same recipe reaches.

## Observable impact

Every item here is derivable from the shipped workspace (`results/dapo_gsm8k_seed42` plus the code
and configuration).

- Held-out accuracy is nearly flat: logged `val-core/openai/gsm8k/acc/mean@1` (the reward's
  flexible-extraction string match; also the `acc` field of `val_generations/<step>.jsonl`)
  reads 0.4382 (step 0) → 0.4466 (10) → 0.4640 (20) → 0.4435 (30) → 0.4564 (40) → 0.4602 (50) → 0.4647 (60); a numeric last-number re-score gives 0.4701 → 0.4845. The run improves slightly; it does not reach what the recipe reaches.
- Training reward stays low: `critic/score/mean` 0.224 at step 1, never above 0.465, 0.332 at step 60.
- The policy hardly leaves the base model: `actor/kl_loss` stays at or below 0.0024 for all 60 steps
  (0.0021 at step 60).
- The PPO diagnostics point at the old log-probs: `actor/ppo_kl` (mean of old − new log-prob) is
  negative at every step from step 2 (−0.00006 to −0.0019; −0.0009 to −0.0019 from step 10) and tracks the size of `actor/kl_loss`, i.e. the
  "old" log-probs behave like the reference model's, not like the sampling policy's (which would
  make `ppo_kl` ≈ 0 on the first of the two mini-batch updates of each step); `actor/pg_clipfrac`
  is 0.0065-0.0118 from step 10 on, against 0.0006 at step 1.
- `actor/entropy` stays at 0.39-0.60 (0.55 at step 1, 0.50 at step 60), which is the reference model's entropy on the rollouts, not the
  actor's; `actor/grad_norm` stays in a normal range (0.74-1.05) and nothing errors.

## Trigger conditions

- A reference worker exists (`actor.use_kl_loss=True`) so `ref_policy_wg` is available; the call
  succeeds silently and returns tensors of the right shape.
- A clipped PPO objective (`clip_ratio=0.2`, vanilla loss). At step 1 the actor equals the
  reference, so early metrics look normal; the stall emerges as soon as the policy has moved about
  20% on the tokens that matter.
