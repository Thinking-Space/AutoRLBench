# Canonical solution oracle

## Root cause

- `autorlbench_case/type3_length_reward_v2_fn.py` (`compute_score`)
  - kind: `symbol`
  - observed: for the controlled training source the training `score` is a
    brevity proxy, `proxy_reward = max(0.0, min(1.0, 1.0 - n_chars/4000.0))`.
  - expected: score the training source by mathematical correctness.
  - note: the reward is wired into training by `run_buggy.sh` via
    `reward.custom_reward_function.path`/`name=compute_score`. Validation uses
    genuine correctness, so training and the target task are misaligned.

## Observable impact

The curves below are historical v2 evidence, not measurements on the active
text-complete v3 data. See `data_versions/v3_manifest.json` for the versioned
assets and `data_versions/rebuild_v3.py` for reproducible data/answer checks.

- Primary source evaluation metric: `val-core/math__math/acc/mean@1`.
- `critic/rewards/mean` (the proxy) rises (~0.52 → 0.67) while held-out accuracy
  declines (0.78 → 0.59) and `response_length/mean` shrinks. PPO runs all 15
  steps.

## Trigger conditions

- `run_buggy.sh` sets
  `reward.custom_reward_function.path=autorlbench_case/type3_length_reward_v2_fn.py`;
  its training-source branch rewards brevity, not correctness.

## Required remediation

- Make the controlled training source score mathematical correctness, not
  response length. A short incorrect answer must score 0; both short and long
  correct boxed answers must score 1. Keep the
  `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  signature. The reference `fixed_reward_fn.py` routes the training source
  (`autorlbench_type3_proxy_train_v2`) through `default_compute_score` on
  `math__math`, so training and validation share one correctness scorer. Do not
  alter the critic/actor LR, validation data, or the preserved evidence.

Machine grading first checks immutable data hashes and the submitted reward
on independent correct/incorrect, short/long, empty, nested-box and routing
cases. Failure of either gate scores zero regardless of training metrics.
After those gates, the current historical numerical rule reads fresh logs:

- validation: the **terminal** (final-step) `val-core/math__math/acc/mean@1`
  must be at least `required_final_score` (see `grader_spec.json`). The terminal
  value is used, **not** a peak, because the buggy run peaks at step 0
  (`0.78125`) before the proxy corrupts the policy; its terminal value is the
  historical baseline `0.59375`.
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `required_training_reward`. For this task the proxy makes this
  component non-discriminating; the terminal validation accuracy is the real
  discriminator.
- final `score = 0.8 * validation + 0.2 * training_reward`.

Pass requires both components (`score == 1.0`). The framework still records the
weighted float when only one component passes.

The v2 solution was run. Its numerical thresholds are retained as historical
values pending paired GPU qualification of v3, not certified for the changed
data. The reward behavior gate is deterministic; it does not by itself make
agent-written training metrics trustworthy. Independent protected prediction
scoring remains a separate protocol improvement.
