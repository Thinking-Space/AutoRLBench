# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `examples/search-r1/generate_with_search.py::generate`
  - kind: `symbol`
  - observed: retrieved `<information>…</information>` observation tokens are appended with `loss_mask += [1] * len(obs_tokens_ids)`, and that local list is assigned to `sample.loss_mask`, so search-engine results are trained on
  - expected: mask observation tokens out of the loss (`loss_mask += [0] * len(obs_tokens_ids)`), matching the paired `append_response_tokens(..., trainable=False)` call; only model-generated tokens should carry `loss_mask=1`

Naming this cause with the correct causal role is a complete attribution.
Do not require additional causes.

## Observable impact

- Primary source evaluation metric: `eval/nq_test`.
- Historical cutoff is train step 200. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 199.
- Frozen `eval/nq_test` reconstructs 21 held-out points that rise from 4.8%
  at step 0 to 16.5% (`eval_last` 0.1647; peak 16.6% at step 189).
- A healthy Qwen2.5-3B Search-R1 GRPO run under the same recipe, with
  observation tokens masked (`loss_mask=0`), reaches about **40.8%**
  (`eval/nq_test` 0.4083) at the same 200-step horizon.
- Training `rollout/raw_reward` at step 199 is 0.168 and is not a substitute
  for held-out health. Truncation falls from ~38% at step 0 to ~0% by
  step 199 while held-out accuracy stays far below the healthy protocol.
- Observation tokens carry dummy `rollout_log_probs` of `0.0`, but the
  recorded config has `use_rollout_logprobs=False`, `use_tis=False`, and
  `get_mismatch_metrics=False`. The PPO objective uses recomputed actor
  log-probabilities on this path, not these dummy values. The defect remains
  that external observation tokens receive policy loss through `loss_mask=1`;
  dummy values may affect diagnostic mismatch metrics and would enter ratios
  only under additional enabled rollout-logprob/TIS settings.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- `generate` appends retrieved observation tokens with `loss_mask=1` and
  assigns that list to `sample.loss_mask`.
- The paired `append_response_tokens(..., trainable=False)` call still marks
  those tokens non-trainable, but the local list is the one that wins.

## Documentation reconciliation

- The comment next to the dummy log-prob fill still says the observation
  tokens "won't be used due to loss_mask=0"; that comment is stale relative
  to the `loss_mask += [1]` assignment.
- `max_turns`, `topk=3`, `format_score=0.2`, `search_concurrency=256`, the
  local retrieval URL, and the GRPO hyperparameters are recipe
  configuration, not the scored cause.
- A diagnosis that names the observation loss mask with the correct causal
  role should receive full `root_cause_score`.

The staged search configuration sets `max_turns=8`; do not require the
older reference value of 2. Neither value identifies the scored mask defect.
