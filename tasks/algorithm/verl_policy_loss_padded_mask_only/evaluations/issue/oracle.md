# Canonical diagnosis

Expected verdict: **unhealthy**.

## Root cause

- `workspace/case/loss.py`, `compute_policy_loss` (registered as `luspo_token`;
  selected by `actor.policy_loss.loss_mode: luspo_token` with
  `actor.loss_agg_mode: seq-mean-token-sum` in `workspace/case/recorded_config.yaml`): `loss = (per_token_loss * mask).sum(-1).mean()` sums the valid token
  losses per response and averages over responses, instead of the standard token-mean aggregation;
  the custom registration also omits native dual clipping
  - kind: `algorithm` deviation from the declared standard-GRPO target
  - observed: a coherent, correctly masked objective — not an intrinsically invalid one — but not
    the target this benchmark declares (`benchmark_contract.md`).
- `workspace/case/recorded_config.yaml`, `actor_rollout_ref.model.use_remove_padding: false`
  - kind: `configuration` deviation from the reference
  - observed: the shared repaired reference runs with `use_remove_padding: true`, so this run
    differs from the reference by both the custom loss and the disabled unpadding.
  - expected: `use_remove_padding: true` as in the reference.

For this variant the difference from the standard reference is composite: naming only the custom
loss or only the disabled unpadding is partial attribution; naming both with their correct causal
roles is complete.

## Observable impact

- `actor/grad_norm`: 267 (step 1) → maximum 1.5e8 → 8.2e5 (step 150); `actor/luspo_token_loss`
  reaches 1.9e5.
- Training reward still rises: `critic/score/mean` 0.095 (step 1) → 0.316 (step 30), peak 0.328
  (step 46), 0.247 (step 150).
- AIME24/25 mean@32 at step 150: 0.057 (0.047 / 0.068), versus 0.144 for the shared repaired
  reference.
- `actor/padding_fraction` runs 0.82 → 0.63; `actor/entropy` falls 212 → 62.

## Documentation reconciliation

- "The loss ignores the response mask" is wrong: the mask is applied; the deviation is the
  sequence-sum-then-average aggregation and the missing dual clipping relative to the declared
  target.
- "Zero gradients / no learning" is wrong: grad_norm reaches 1.5e8.
- Naming only the loss or only the unpadding setting earns partial, not full, root-cause credit:
  the reference differs from this run in both.
