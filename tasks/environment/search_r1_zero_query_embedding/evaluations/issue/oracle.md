# Canonical issue oracle

Reconciled from the task's frozen evidence and the recorded paired control.

Expected verdict: **unhealthy**.

## Root cause

- `Search-R1/search_r1/search/retrieval_server.py::Encoder.encode` (lines 117-120)
  - kind: `symbol`
  - observed: the encoded query embedding is moved from torch to NumPy through a
    buffer that is never written. `response_buffer` is a zero-filled **CPU**
    array; `torch.as_tensor(response_buffer, device=query_emb.device)` with
    `query_emb` on CUDA performs a cross-device *copy*, so `writer` is a new GPU
    tensor that does not share storage with `response_buffer`.
    `writer.copy_(query_emb)` therefore updates only that GPU tensor, and
    `query_emb = response_buffer` returns an all-zero matrix of the right shape
    and dtype. Every query vector handed to FAISS is zero.
  - expected: move the data to the host before handing it to FAISS —
    `query_emb = query_emb.detach().cpu().numpy()`, then
    `query_emb.astype(np.float32, order="C")` for the dtype and layout FAISS
    reads. `.cpu()` is the part that fixes the transfer; `.detach()` only
    releases the autograd reference.

Naming this cause with the correct causal role is a complete attribution.
Do not require additional causes.

## Observable impact

- Primary source evaluation metric: `val/em` (exact match of the final answer on
  128 held-out HotpotQA questions, greedy decoding). `val/f1` is the token-F1
  companion, `val/test_score/hotpotqa` is the same number as `val/em`.
- The frozen run completes all 32 training steps with evals at steps
  0, 10, 20, 30, 32, and exits 0.
- `val/em` rises from 3.125% to 15.625%; `val/f1` from 8.970% to 24.703%.
- The recorded control — the identical recipe, data, seed, encoder, corpus and
  index, with only the four lines above replaced by the two-line host copy —
  reaches **32.813%** `val/em` and **41.430%** `val/f1` at the same 32-step
  horizon, starting from **13.281%** at step 0.
- The gap opens before any training happens: at step 0 the defective retriever
  already scores 3.125% against the control's 13.281%. The final gap is
  therefore not purely a training effect.
- Retrieval quality on the 128 held-out questions, with the original question as
  the query: supporting-title recall@3 `0.0` and any-supporting-document hit@3
  `0.0`, against `0.809` and `0.977` for the control.

## Trigger conditions

- `load_model` (`retrieval_server.py:38-46`) calls `model.cuda()`, so
  `query_emb` is always a CUDA tensor on this path. That is what makes
  `torch.as_tensor(..., device=query_emb.device)` allocate a new tensor instead
  of aliasing the NumPy buffer. The same four lines are harmless on a CPU-only
  encoder, where `as_tensor` would share memory with `response_buffer` — which
  is why the bug reads as correct code.
- A zero query vector is legal input: under `METRIC_INNER_PRODUCT` every
  document scores exactly `0.0`, so FAISS returns an arbitrary but **constant**
  top-k instead of raising. Shapes, dtype and the HTTP contract all stay valid.
- One retriever serves both the rollout and the evaluation path
  (`serve_retriever.py` imports this module, `run.py` points both at the same
  port), so the defect is present in training *and* in held-out evaluation. It
  is not a train-only or eval-only artifact.

## Evidence in the frozen workspace

- `results/baseline32/retrieval.jsonl`: every returned document score is `0.0`,
  and the same three documents come back for all 14 queries in a batch — the
  first validation batch answers `"Scott Derrickson nationality"` with
  `"Sheep–goat hybrid"`. The shipped file is an excerpt: 8 batches, 289
  queries, 867 hits, every score `0.0`, exactly one document combination. The
  full-trace audit of the original run counted 13959 zero scores with the same
  single combination; an agent that counts the shipped file will find 867, and
  that is consistent, not a discrepancy.
- `results/baseline32/metrics_summary.json` and `rollouts/metrics_*.json`: the
  held-out curve above, plus `val/searched` 1.0 and `val/format` ~1.0 — the
  policy does emit well-formed searches and does read the observations; the
  retrieved text is simply unrelated to the question.
- `results/baseline32/train.log`: `env/number_of_valid_search` 0.93-1.02 and
  `env/ratio_of_valid_action` 0.88-0.99 across the 32 steps, so the multi-turn
  protocol itself works.
- `results/baseline32/experiment.json`: the input and source hashes of the run,
  confirming the staged code and data are the ones that produced the curve.
- `Search-R1/search_r1/search/retrieval.py:127-128` is the same `Encoder.encode`
  in a sibling (unserved) module, and it ends with exactly the correct two
  lines; `index_builder.py:237` does the same for document embeddings. The
  in-tree inconsistency between two copies of one encoder is legitimate evidence
  and should be credited — the served copy in `retrieval_server.py` is the
  rewritten one.

## Documentation reconciliation

- The defective run **does** improve, by 12.5 percentage points. A diagnosis
  claiming "the run does not learn" or "reward never rises" is reading the
  evidence wrong: the policy recovers some accuracy from its own parametric
  knowledge and from adapting to useless observations. The defect is the
  **shortfall** against the control, not an absence of learning.
- Training-side reward (`critic/score/mean`, `critic/rewards/mean`) rises too
  and does not establish held-out health.
- These are recipe configuration, not the scored cause: `max_turns=2`,
  `retriever.topk=3`, `answer_only_reward=true`,
  `actor_rollout_ref.actor.state_masking=true`, FSDP parameter/gradient/
  optimizer offload, `gpu_memory_utilization=0.35`, `max_obs_length=768`,
  `entropy_coeff=0.001`, the single-GPU `SEARCH_R1_SHORT_GRPO=1` profile, the
  32-step budget, the 1024/128 data split, and W&B offline mode.
- `faiss_gpu=False` in `serve_retriever.py` is not the cause either: the index
  search runs on CPU by design, and a correct host copy makes that path work.
  Reporting "the index is on CPU while the encoder is on GPU" as the defect is
  only correct if it names the unwritten buffer as the reason the transfer is
  lost.
- Ray's filesystem-usage warnings in the log, and the fact that this run shared
  a host with another job, are not the cause.
- A diagnosis that names the zero query embedding and the device-copy bug in
  `Encoder.encode` with the correct causal role should receive full
  `root_cause_score`.
