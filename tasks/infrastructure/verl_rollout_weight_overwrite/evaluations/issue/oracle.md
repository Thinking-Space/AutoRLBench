# Canonical diagnosis

Expected verdict: **unhealthy**.

## Root cause

- `workspace/case/sync.py`, `finalize_weights` reloads the initial safetensors checkpoint into the
  vLLM engine after Actor weights have been synchronized (the hook also sits on the async update
  path)
  - kind: `infrastructure` implementation defect
  - observed: after each Actor-to-vLLM weight sync, the hook replaces the just-synced weights with
    the initial checkpoint, so rollout generation repeatedly uses the initial weights. The Actor
    keeps training against its own updated weights, so the policy drifts away from what rollout
    samples.
  - expected: the post-sync hook is telemetry-only — it records the fingerprint diff
    (`sync_trace.jsonl`) without loading any checkpoint.

Naming the post-sync initial-checkpoint reload with its correct causal role (rollout engine reset
to initial weights → generation never uses the updated Actor → training signal computed against
stale samples → flat reward and frozen held-out accuracy) is a complete attribution. Do not require
additional causes.

## Observable impact

- `sync_trace.jsonl` (2224 records): `changed_parameters` is 0 in the early records and rises to
  154–156 of 226 sampled parameters, with `after_sync_max_abs_delta` up to 4.9e-4 — the reload
  actively overwrites drifted weights.
- `actor/grad_norm` is nonzero and grows 0.24 → maximum 83.7 (46.3 at step 150): Actor updates
  happen.
- `actor/ppo_kl` grows to 1.38 (0.69 at step 150) and `actor/pg_clipfrac` to 0.083, while
  `critic/score/mean` stays flat around 0.08–0.12: the policy moves but rollout never follows.
- AIME24/25 mean@32 at step 150: 0.0125; `val-core/aime24/acc/mean@32` is exactly equal at step 0
  and step 150 (0.0229) — held-out accuracy stays frozen at the step-0 level.
- `critic/advantages/mean` is nonzero (about −0.013; extrema about +2.5/−2.5): this is not a
  zero-advantage failure.

## Documentation reconciliation

- "Zero gradients / no learning" is wrong: grad_norm, ppo_kl and clipfrac all grow.
- "The reward callback broadcasts one score per batch" is wrong: advantages are nonzero here; that
  is a different failure with a different signature.
- Blaming vLLM, FSDP, the async engine or the learning rate alone is wrong: the sync path delivers
  the updated weights and the hook then overwrites them; the trace localizes the reset.
