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

Machine grading reads only the new experiment logs (200-step budget). Scoring:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.10`
  scores `1`, otherwise `0`. (Repaired reference run at the 200-step budget:
  peak `0.114` at step 100, `0.103` at step 200, from a step-0 baseline of
  `0.049`; the broken run's best is `0.076` at step 50 and `0.0` from step
  150 on.)
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `0.10` to score `1`, otherwise `0`. (Repaired reference
  run: mean `0.176` over steps 1–200; the broken run: `0.065`.)
- final score is binary (`1.0` / `0.0`): the validation component decides
  whenever the fresh run logged any validation points; the training-reward
  component applies only as a fallback for runs that produced no validation
  metric at all.
