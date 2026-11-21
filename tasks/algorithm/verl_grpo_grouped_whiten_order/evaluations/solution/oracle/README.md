# Canonical solution oracle

## Root cause

- `verl/utils/torch_functional.py::grouped_whiten` (and `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`)
  - kind: `symbol`
  - observed: `grouped_whiten` returns `torch.gather(whitened, 0, order)` using the forward sort
    index instead of the inverse scatter permutation, misassigning per-prompt standardized
    advantages across non-contiguous prompts.
  - expected: compute per-UID mean/std directly in `compute_grpo_outcome_advantage` (or scatter
    `whitened` back out-of-place via `torch.empty_like(whitened).scatter_(0, order, whitened)`).

- `verl/trainer/ppo/core_algos.py::kl_penalty`
  - kind: `symbol`
  - observed: replaces `if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):` with
    `if kl_penalty in ("kl", "k1", "abs", "mse", "k2"):`, routing `low_var_kl` into a straight-through
    estimator whose `backward_score = 0.5 * seq_log_ratio.square().expand_as(logprob)` squares the
    sequence-summed log-ratio and broadcasts across tokens, preserving forward `actor/kl_loss` while
    inflating backward KL regularization by ~`L_i` per token (~`L_i^2` per sequence).
  - expected: return `forward_score` when `not kl_penalty.endswith("+")` and use per-token
    `backward_score = 0.5 * (logprob - ref_logprob).square()` for `+`-suffixed variants.

## Observable impact

- Primary source evaluation metric: `heldout/gsm8k_last_number_acc@50` ends at `0.4610` / `0.4640`
  (logged `val-core/openai/gsm8k/acc/mean@1`: `0.4337` / `0.4375`, `actor/kl_loss`: `0.00795` /
  `0.00635`) versus `0.5588` / `0.5368` (`actor/kl_loss`: `0.05953` / `0.08392`) in the control.

## Trigger conditions

- Same as `evaluations/issue/oracle.md`.

## Required remediation

- `verl/trainer/ppo/core_algos.py` & `verl/utils/torch_functional.py`: restore direct per-UID
  standardization in `compute_grpo_outcome_advantage` (removing `grouped_whiten`, or inverting its
  permutation via out-of-place `torch.empty_like(whitened).scatter_(0, order, whitened)`), and
  restore `if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):` plus per-token
  `backward_score = 0.5 * (logprob - ref_logprob).square()` in `kl_penalty` (`fix.diff`).

Machine grading checks only the fresh rerun logs. It re-scores `experiment_logs/val_generations/50.jsonl`
(`last_number` rule, `>= 1,300` rows, pass at `>= 0.51`), verifies continuous `metrics.jsonl` rows across
steps `0..50` with `val-core/openai/gsm8k/acc/mean@1` at step 50, and checks `run_grpo_gsm8k.sh`,
pinned files, and generated parquets. Fixing only `kl_penalty` drops step-50 accuracy to `0.4102` /
`0.3654 < 0.51`, while fixing only `grouped_whiten` holds it at `0.4807` / `0.4905 < 0.51`.
