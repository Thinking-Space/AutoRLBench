# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`
  - kind: `symbol`
  - observed: the labels for the log-prob recompute are built un-rolled —
    `input_ids_rmpad_rolled = input_ids_rmpad` (the `# for compute the log_prob` line)
  - expected: the causal roll `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad,
    shifts=-1, dims=1)`, so label position `i` holds `token[i+1]`, the next-token
    target that the logit at position `i` predicts
  - note: `input_ids_rmpad_rolled` feeds both `logprobs_from_logits` and the
    fused-kernel `shift_labels`. `logprobs_from_logits` does a plain
    log-softmax+gather with no internal shift, and the model's logits are standard
    (`logits[i]` → `token[i+1]`), so dropping the roll misaligns every recomputed
    response log-prob by one position. The `verl/workers/utils/padding.py::
    no_padding_2_padding` left-shift slice is the correct consumer of a *rolled*
    array and is NOT the defect.

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- Training starts and computes a finite actor loss, but the recomputed token-level
  log-probs are shifted by one position relative to the rollout.
- The train/rollout mismatch explodes (`rollout_corr/kl` ~0 → O(10),
  `rollout_corr/ppl_ratio` ~1.0 → ~1e10), corrupting GRPO updates so that held-out
  TACO accuracy and training reward collapse to 0 and response length balloons.

## Trigger conditions

- `prepare_model_inputs` assigns `input_ids_rmpad_rolled = input_ids_rmpad`,
  dropping the `torch.roll(input_ids_rmpad, shifts=-1, dims=1)`.
- Tensor shapes are unchanged, so `logprobs_from_logits` / `cross_entropy` do not
  crash — the only symptom is the corrupted alignment and the metric collapse.

## Documentation reconciliation
- Launch hyperparameters, TACO parquet, and the Qwen2.5-1.5B-Instruct checkpoint
  are fixed and not the defect.
- The downstream `no_padding_2_padding` left-shift slice is healthy; the missing
  piece is the upstream causal roll in the engine, not the padding slicer.
