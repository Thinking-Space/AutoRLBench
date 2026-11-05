# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

This run carries two independent defects; a complete attribution names both.

- `examples/search-r1/generate_with_search.py::generate`
  - kind: `symbol`
  - observed: retrieved `<information>…</information>` observation tokens are appended with `loss_mask += [1] * len(obs_tokens_ids)`, and that local list is assigned to `sample.loss_mask`, so search-engine results are trained on
  - expected: mask observation tokens out of the loss (`loss_mask += [0] * len(obs_tokens_ids)`), matching the paired `append_response_tokens(..., trainable=False)` call; only model-generated tokens should carry `loss_mask=1`

- `examples/search-r1/generate_with_search.py::_passages2string`
  - kind: `symbol`
  - observed: each retrieved document is rendered as `Doc {idx+1}(Title: {title})\n` — the passage body `text` (split off the document `contents` right above) is never appended, so every `<information>` observation carries document titles only and the policy never sees retrieved passage text
  - expected: `Doc {idx+1}(Title: {title}) {text}\n`, so the observation includes the passages the retriever matched; with title-only observations the search channel is hollowed out on the input side, independently of the loss-side defect

Naming both causes with their correct causal roles is a complete
attribution; naming only one of the two is partial. Do not require
additional causes beyond these two.

## Observable impact

- Primary source evaluation metric: `eval/nq_test`.
- Historical cutoff is train step 200. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 199.
- Frozen `eval/nq_test` reconstructs 21 held-out points that rise from 3.3%
  at step 0 to 16.7% (`eval_last` 0.1673; peak 16.8% at step 179).
- A healthy Qwen2.5-3B Search-R1 GRPO run under the same recipe, with
  observation tokens masked (`loss_mask=0`), reaches about **40.8%**
  (`eval/nq_test` 0.4083) at the same 200-step horizon.
- Training `rollout/raw_reward` at step 199 is 0.1680 and is not a substitute
  for held-out health. Truncation falls from ~38% at step 0 to ~0% by
  step 199 while held-out accuracy stays far below the healthy protocol.
- Observation tokens also carry dummy `rollout_log_probs` of `0.0`, so
  `loss_mask=1` on those tokens optimizes against meaningless reference log
  probs and corrupts the GRPO objective.
- The rendering defect is code-visible in the same file:
  `_passages2string` computes the passage body
  (`text = "\n".join(content.split("\n")[1:])`) and then drops it from the
  formatted string, so the `<information>` block the policy sees — and that
  the loss-mask defect trains on — is titles only.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- `generate` appends retrieved observation tokens with `loss_mask=1` and
  assigns that list to `sample.loss_mask`.
- The paired `append_response_tokens(..., trainable=False)` call still marks
  those tokens non-trainable, but the local list is the one that wins.
- `_passages2string` builds the `<information>` observation from document
  titles only (the passage body is split off but never appended), while the
  surrounding retrieval pipeline (top-k lookup, local search URL,
  concurrency) is otherwise intact.

## Documentation reconciliation

- `max_turns`, `topk=3`, `format_score=0.2`, `search_concurrency=256`, the
  local retrieval URL, and the GRPO hyperparameters are recipe
  configuration, not scored causes.
- A diagnosis that names both the observation loss mask and the title-only
  retrieval rendering, each with its correct causal role, should receive
  full `root_cause_score`; naming only one of the two is partial.
