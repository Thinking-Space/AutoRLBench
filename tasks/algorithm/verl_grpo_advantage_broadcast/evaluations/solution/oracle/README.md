# Canonical solution oracle

## Root cause

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
  - kind: `symbol`
  - observed: the length-`B` GRPO advantages are laid along the sequence axis and expanded across rows, then multiplied by `response_mask`. Shape stays `[B, S]`.
  - expected: `scores.unsqueeze(-1) * response_mask`

## Observable impact

- With batch 16 and rollout `n=8`, `B=128` while the response length is 4096, so most tokens get advantage 0 and early positions share one advantage vector across the batch.
- Training does not crash, because the returned shape is still `[B, S]`.
- The bundled run (stopped at step 284) keeps `critic/rewards/mean` near 0.12 while held-out accuracy stays at 0.043-0.076.

## Trigger conditions

- `adv_estimator=grpo` uses `compute_grpo_outcome_advantage`, not the vectorized estimator.

## Required remediation

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`: `restore`; target state/value: `scores = scores.unsqueeze(-1) * response_mask` after the per-uid normalization.


Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.10`
  (10%). The bundled defective run's last validation (step 250) is `0.0543`
  and its peak is `0.0761` (step 0).
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `0.14`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.
