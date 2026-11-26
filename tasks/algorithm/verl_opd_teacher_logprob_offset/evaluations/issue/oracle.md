# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and a
recorded aligned run (same code, data, teacher, seed and budget; only
`verl/workers/rollout/vllm_rollout/utils.py::extract_prompt_logprobs` differs).

Expected verdict: **unhealthy**.

## Root cause

- `verl/workers/rollout/vllm_rollout/utils.py::extract_prompt_logprobs`
  - kind: `symbol`
  - observed: the function starts the lists with a zero placeholder for the first prompt token
    (`prompt_logprobs_ls = [[0.0] * ...]`) and no longer appends the trailing dummy entry. Entry `i`
    therefore holds the teacher's log-prob of token `i` instead of token `i+1`, one position off the
    student's next-token log-probs it is compared with in the distillation loss. The per-token
    log-ratio used as the (negative) advantage compares log-probs of different tokens, so the
    update direction is noise from step 1.
  - expected: upstream's layout — skip vLLM's `None` entry for the first prompt token and pad one
    dummy entry at the end, so teacher and student log-probs are aligned token by token.

Naming the off-by-one in `extract_prompt_logprobs` with its causal role (shifted teacher log-probs →
the distillation advantage compares different tokens → noisy, large updates → the student's
distribution collapses to near-uniform and accuracy goes to 0) is a complete attribution. Do not
require additional causes.

**The shifted teacher log-probs in `extract_prompt_logprobs` are required.** "Distillation is
unstable", "the teacher signal is wrong" or "teacher and student are misaligned" without locating
the extraction code earns at most partial root-cause credit. Blaming the learning rate, clipping,
the k1 estimator or the policy-gradient formulation, teacher quality, chat templates or tokenizers,
response length, or the task reward is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run60` plus the code and
configuration).

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` 0.432 (0) → 0.031 (10) → 0.0
  (20-60); re-scored from `val_generations` by last number: 0.4663 (0), 0.0622 (10), 0.0000 (20),
  0.0000 (30), 0.0000 (40), 0.0000 (50), 0.0000 (60).
- Step 1, before any update: `actor/distillation/abs_loss` 0.72, `actor/distillation/loss_min`
  -15.8 / `loss_max` 20.2, `actor/grad_norm` 22.8; `abs_loss` rises to 3.2 by step 5 and
  `grad_norm` to 66-73 at steps 5-10.
- `actor/entropy` 0.49 (1) → 1.39 (3) → 4.48 (5) → 9.36 (10) → 11.92 (60), the entropy of a uniform
  distribution over the 151,936-token vocabulary.
- `response_length/mean` 299 → 505 by step 10 with clip ratio 0.97-1.00 afterwards; rollouts and
  validation responses at step 60 are degenerate strings (e.g. `oldem..,.,.,.,`).
- `critic/score/mean` (task reward, logged but unused) 0.41 at step 1, 0.02 at step 5, 0.0 from step
  20.

## Trigger conditions

- Teacher prompt log-probs from vLLM (`prompt_logprobs`) consumed by verl's distillation loss; any
  shift of the returned list against the student's next-token layout corrupts every token's
  log-ratio while keeping the shapes valid.

## Documentation reconciliation

- The comment above the changed lines describes the new layout as an alignment fix; the student
  log-probs it is compared with use the next-token layout, so the comment is the defect's
  justification, not evidence against it.
