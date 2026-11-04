# Canonical solution oracle

## Root cause

- `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`
  - kind: `symbol`
  - observed: the per-sequence labels used for the log-prob recompute are built
    un-rolled — `input_ids_rmpad_rolled = input_ids_rmpad` (the `# for compute the
    log_prob` line)
  - expected: the causal roll `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad,
    shifts=-1, dims=1)`, so label position `i` holds `token[i+1]` (the next-token
    target the logit at position `i` predicts)
  - note: this is the live FSDP/NO_PADDING log-prob path. `input_ids_rmpad_rolled`
    is passed both to `logprobs_from_logits` and to the fused-kernel `shift_labels`.
    `logprobs_from_logits` does a plain log-softmax+gather (no internal shift), and
    the Qwen2.5 logits are standard (`logits[i]` predicts `token[i+1]`), so without
    the roll every recomputed response log-prob is misaligned by one position. The
    downstream `verl/workers/utils/padding.py::no_padding_2_padding` left-shift
    slice (`values[seq_offset - resp_len - 1 : seq_offset - 1]`) is the correct
    *consumer* of a rolled array and is NOT the defect — it cannot recover the
    alignment when the upstream roll is missing.

## Observable impact

- Primary source evaluation metric: `val-core/taco/acc/mean@1`.
- Loss stays finite (shapes are unchanged), so training does not crash.
- The train-vs-rollout log-prob mismatch explodes: `rollout_corr/kl` jumps from
  ~0 to O(10) and `rollout_corr/ppl_ratio` from ~1.0 to ~1e10. GRPO updates are
  corrupted, so held-out TACO accuracy and training reward collapse to 0 and the
  response length balloons.

## Trigger conditions

- `prepare_model_inputs` is observed assigning `input_ids_rmpad_rolled =
  input_ids_rmpad`; the canonical expectation is
  `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`.

## Required remediation

- `verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`: `restore`;
  target state/value:
  `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`.

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.10`
  scores `1`, otherwise `0`. (Healthy reference run: peak `0.125` at step 150,
  final `0.109`, from a step-0 baseline of `0.054`; the broken run collapses to
  `0.0`.)
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `0.10` to score `1`, otherwise `0`. (Healthy reference run:
  mean ≈ `0.15`–`0.20`; the broken run collapses to ~`0.02`.)
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.
