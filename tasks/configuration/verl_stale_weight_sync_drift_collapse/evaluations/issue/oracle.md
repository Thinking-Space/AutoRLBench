# Canonical issue oracle

Reconciled from the task's frozen evidence, the control and severity-ladder
runs (per-step metrics in `evaluations/issue/reference/`), and the 120-step
diagnostic long run.

Expected verdict: **unhealthy**.

## Root cause

- `run_grpo.sh`: `trainer.rollout_sync_interval=64` (the throughput
  experiment knob in `verl/trainer/config/ppo_trainer.yaml`, consumed by the
  gated `checkpoint_manager.update_weights()` call in
  `verl/trainer/ppo/ray_trainer.py`)
  - kind: `hparam` (configuration interacting with framework behaviour)
  - observed: the rollout replicas stay resident
    (`actor_rollout_ref.rollout.free_cache_engine=False`) and the actor's
    weights are only pushed to them on multiples of 64 optimization steps, so
    every rollout batch between syncs is sampled from a policy up to 63 steps
    stale. In this verl the trainer recomputes `old_log_probs` from the
    **current** actor weights after generation, so the PPO/GRPO ratio and
    clip metrics only measure within-step policy movement — they stay benign
    and never see the sampler-trainer mismatch. The mismatch is real: with
    the sampler `q` (stale weights) and the scores `s_p` from the current
    policy `p`, the expected update `E_q[R · s_p]` splits into
    `E_q[R | prefix] · s̄(prefix) + Cov`; on-policy the drift term is zero
    (scores of the generating policy sum to zero per prefix), off-policy it
    is a non-vanishing distillation-toward-the-stale-sampler gradient that
    compounds between syncs. With the sync deferred 64 steps the drift
    dominates the learning signal for the entire run: the training reward
    never leaves its initial 0.02–0.05 band (vs 0.26–0.31 for the every-step
    control at step 70), the entropy decays 0.69 → 0.07, and the first
    mid-run sync at step 64 replaces the sampler with the drifted,
    entropy-collapsed trainer policy — from step 65 the reward is exactly
    0.0 with responses pinned to the 512-token cap.
  - expected: `rollout_sync_interval=1` (per-step sync; the only difference of
    the healthy control), a smaller interval, or an off-policy correction aimed
    at the sampler-trainer mismatch (importance weighting against the sampler
    such as verl's `rollout_correction` with `calculate_log_probs=True`, or
    score centering).

A full-credit diagnosis names the reduced weight-sync frequency as the cause
**and** the sampler-trainer staleness mechanism, in three parts. Score semantic
equivalence, not verbatim phrasing — the equivalences below all count:

1. **Periodicity.** Rollout tokens come from a policy many steps behind the
   trainer (the sync itself works, it is just deferred).
2. **Blind ratio/clip anchor.** The trainer recomputes `old_log_probs` from the
   current actor, so the ratio/clip metrics stay blind to the sampler-trainer
   mismatch. Equivalents that count: "the PPO clip only bounds within-step
   drift, never the cumulative divergence from the behavior policy", "the
   importance ratio is near 1 by construction", "the old-policy anchor is the
   current actor, not the sampling policy". Merely noting that no correction
   is applied (`rollout_is=None`, "measured but never compensated") does NOT
   by itself satisfy this part — that is part 3's territory.
3. **Cumulative uncorrected drift.** The mismatch compounds between syncs and
   the update is pulled toward the stale sampling distribution instead of
   task learning. The verbatim phrase "distills toward the stale sampler" is
   NOT required. Equivalents that count: "nothing bounds the cumulative drift
   against the sampling policy", "no closed-loop learning signal — the policy
   cannot be corrected by the consequences of its own changes", "the actor
   over-sharpens onto stale-rollout fragments", "accumulated divergence from
   the behavior policy", "off-policy drift accumulates unchecked", "the
   mismatch is measured (rollout_corr/*) but never applied (rollout_is=None)".

A diagnosis is false on part 3 only if it attributes the damage to a different
mechanism — clip-region exclusion of stale tokens, KL-loss domination, generic
entropy collapse as the initiating cause, or data/reward defects. Naming only
"the rollout does not get the new weights" without the periodicity or without
the cumulative-drift mechanism identifies the right knob but the wrong
mechanism and earns partial `root_cause_score`.

## Observable impact

- Primary sources: `critic/score/mean` (training reward) and
  `training/rollout_probs_diff_mean` (mean per-token |log q − log p| between
  the rollout engine's own logprobs, enabled by
  `actor_rollout_ref.rollout.calculate_log_probs=True`, and the trainer's
  recomputed `old_log_probs`) per step in
  `workspace/results/grpo/metrics.jsonl`; validation accuracy
  (`val-core/.../acc/mean@1`) in the same file; `update_weights done` lines
  in `workspace/results/grpo/train.log`.
- Defective (seed 42, shipped): reward 0.0312 at step 1 and 0.0293 at step
  64 — it never leaves the initial band; `rollout_probs_diff_mean` climbs
  monotonically 0.0046 → 0.0482@15 → 0.0945@30 → 0.2000@45 → 0.5312@64
  (~115× the control's noise floor); entropy decays 0.689 → 0.073; from step
  65 the reward is exactly 0.0000 with response length pinned at 510–511
  (the 512 cap), entropy 0.025, `actor/pg_clipfrac` 0.0000 and
  `actor/grad_norm` ~0.02. Held-out accuracy never moves from its initial
  0.0098–0.0107 and is 0.0000 at step 70.
- `actor/pg_clipfrac` stays at or below 0.03 everywhere and is at 0.000 in the
  collapsed defective run — nothing in the standard PPO/GRPO metrics flags the
  failure.
- Rollout outputs stay coherent through the whole flat phase (step 61 sample:
  `<answer>5 * 8 + 38</answer>`) and only degenerate as part of the policy
  collapse itself (step 66+: arithmetic-word salad degrading into `- - + -
  + -` repetition to the cap) — a statistical drift, not an engine
  corruption from step 1.

Review-side reference arms (NOT in the workspace the diagnosis was made from,
and NOT scoreable as evidence): the seed-43 replication (flat 0.023–0.045
band, probe peak 0.3870@64, reward exactly 0.0000 from step 65, val 0.0000 at
step 70), the every-step control (`SYNC_INTERVAL=1`: reward 0.0430 →
0.2344@45 → 0.2637@70 and val 0.0117 → 0.2725@45 for seed 42; probe stays in
the 0.001–0.005 numerical floor; `update_weights done` every step), the
clipfrac contrast with the control (0.000 vs 0.003), and the k=1/k=8/k=64
severity ladder (k=8: 0.4824 by step 60 then 0.0371@70 with a period-8 probe
sawtooth). These live in `evaluations/issue/reference/`, which only the
evaluator sees. A diagnosis that does not cite them must not be marked false
on that account.

`evidence_score` is computed ONLY on the defective-run core items the
workspace actually contains; all six carry the causal argument and are
scoreable:

1. the flat reward band (steps 1-64) with the exactly-zero reward from step 65;
2. the climbing sampler-trainer divergence metrics — `rollout_probs_diff_mean`
   (0.0046 → 0.5312) OR any equivalent probe from the same family
   (`rollout_corr/kl` 0.0005 → 36.2, `rollout_actor_probs_pearson_corr`
   0.9996 → 0.235, `rollout_corr/training_ppl` blow-up) — with the correct
   rising direction and the post-sync reset;
3. the sync periodicity being countable in the log — exactly two
   `update_weights done` events (initial + step 64), or the
   `timing_s/update_weights` no-op-everywhere-but-step-64 pattern;
4. the entropy decay 0.689 → 0.073;
5. the response length pinned at the 512-token cap from step 65;
6. the held-out accuracy never leaving 0.0098-0.0107 and hitting 0.0000 at
   step 70.

Secondary items (optional, NOT in the denominator): the benign
`actor/pg_clipfrac` contrast (0.000 in collapse) and the
coherent-then-degenerate rollout-text sequence. Citing them is welcome but a
diagnosis that omits them must NOT lose evidence credit on that account.

## Trigger conditions

- verl with the async agent-loop rollout server and resident replicas
  (`free_cache_engine=False`, `mode=async`), i.e. the rollout engine keeps
  its weights between steps.
- `trainer.rollout_sync_interval > 1` — the staged throughput knob that skips
  the per-step weight sync.
- GRPO whose `old_log_probs` are recomputed from the current trainer weights
  (verl default), so no ratio/importance term references the sampler.
- Not the same object as `infrastructure/verl_fsdp_bucketed_weight_sync`: there
  the sync runs every step but drops the first bucket / freezes most
  parameters; here the sync is complete and correct but deferred, and the
  optimizer covers all parameters (the `update_weights done` events in
  `train.log` complete; the probe measures a growing mismatch rather than a
  constant one).
- Not the same object as `configuration/verl_vllm011_fp8_kv_scale_sleep` or
  `infrastructure/verl_quantized_rollout_restart`: no quantization is
  involved, outputs are coherent until the policy itself collapses, and the
  failure is delayed rather than present at step 1.

## Wrong attributions (must not pass)

- "Off-policy data — tighten the clip / lower the LR / fewer epochs": the
  clip metrics are already benign (0.000 in the collapsed phase); the
  PPO-clip regularizes the policy's own movement and cannot see the
  sampler-trainer mismatch (the score-centering literature: PPO/DAPO survive
  64-step staleness in reward-mode terms, but the uncorrected PG/GRPO
  objective itself degrades — clip tightening is not a correction for
  staleness).
- "Advantage normalization on/off": group centering does not remove the
  prefix-conditioned drift term.
- "The rollout never receives the new weights / the sync is broken": the
  initial sync and the step-64 sync complete — `update_weights done` is in
  train.log; the weights flow correctly, just late. Partial credit at most
  without the staleness mechanism.
- "An async framework bug": every component works exactly as configured.
- "LR too low / data or reward problems": the every-step control learns to
  0.25–0.31 training reward and 0.25–0.27 held-out accuracy with the same LR,
  data, and reward.
- "Entropy collapse / exploration issues": the entropy decay is a
  consequence of the drift (distillation toward the stale sampler) and the
  control reaches comparable late-phase entropy (0.016) while holding its
  reward; entropy is a symptom here, not the cause.

## Documentation reconciliation

- The flat reward, the climbing `rollout_probs_diff_mean` (0.0046 → 0.5312),
  the exactly-two `update_weights done` lines (init + step 64), the entropy
  decay, the post-sync reward of exactly 0.0 with length-capped responses, and
  the benign clip metrics are all consistent with the deferred-sync
  staleness story.
- The step-64 sync is not the bug — it is the moment the already-drifted
  trainer policy replaces the init policy in the sampler; the 120-step
  diagnostic run in `reference/long_run_k64_120steps_metrics.jsonl` shows
  the same collapse repeat after that sync (reward ~0.01, entropy 0.006,
  probe ~0.78 at step 120).
- `reference/evidence_summary.txt` tabulates the full severity ladder and the
  k=8 sawtooth; the control and ladder arms there carry documented
  run-time microbatch-packing reductions (OOM avoidance in this colocated
  setup) that do not change the training objective.
