# Canonical diagnosis

Expected verdict: **unhealthy**.

## Root cause

- `workspace/case/loss.py`, `compute_policy_loss` (registered as `luspo_token`;
  selected by `actor.policy_loss.loss_mode: luspo_token` with
  `actor.loss_agg_mode: seq-mean-token-sum` in `workspace/case/recorded_config.yaml`): `loss = (per_token_loss * lengths).mean()` multiplies every token
  position — padded or not — by the full response length and averages the whole padded tensor without
  applying `response_mask`
  - kind: `algorithm` deviation from the declared standard-GRPO target
  - observed: padded positions contribute to the objective, the loss scale grows with response
    length, and the custom registration replaces standard token-mean normalization and native dual
    clipping. With `use_remove_padding: false` the full padded tensor enters the loss directly.
- `workspace/case/recorded_config.yaml`, `actor_rollout_ref.model.use_remove_padding: false`
  - kind: `configuration` deviation from the reference
  - observed: the shared repaired reference runs with `use_remove_padding: true`, so this run
    differs from the reference by both the custom loss and the disabled unpadding.
  - expected: `use_remove_padding: true` as in the reference.

For this variant the difference from the standard reference is composite: naming only the custom
loss or only the disabled unpadding is partial attribution; naming both with their correct causal
roles is complete.

## Observable impact

- `actor/padding_fraction` climbs 0.82 → 0.9955: with unpadding disabled and responses shrinking,
  nearly every position in the loss input is padding.
- `actor/entropy` collapses 210 → 0.80 and `response_length/mean` collapses from about 1500 to
  about 37 tokens (minimum 24.8).
- `actor/grad_norm` starts at 1.7e5 and reaches a maximum of 3.0e7; `actor/ppo_kl` spikes to 0.67.
- Training reward peaks at 0.304 (step 38) and falls back to 0.090 (step 150).
- AIME24/25 mean@32 at step 150: 0.0031 (0.003 / 0.003), versus 0.144 for the shared repaired
  reference.

## Documentation reconciliation

- "Zero gradients / no learning" is wrong: grad_norm reaches 3.0e7.
- "The reward callback broadcasts one score per batch" is wrong: that is a different failure with a
  different signature; advantages here are nonzero.
- Naming only the loss or only the unpadding setting earns partial, not full, root-cause credit:
  the reference differs from this run in both.
