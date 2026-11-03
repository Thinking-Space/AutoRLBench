# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `autorlbench_case/type3_length_reward_v2_fn.py::compute_score`
  - kind: `symbol`
  - observed: for the controlled training source (`data_source == "autorlbench_type3_proxy_train_v2"`) the returned training `score` is a brevity proxy, `proxy_reward = max(0.0, min(1.0, 1.0 - n_chars/4000.0))`; shorter responses score higher regardless of correctness
  - expected: score the training source by mathematical correctness — a short incorrect answer scores 0; both short and long correct boxed answers score 1; response length must not determine the training reward
  - note: this reward is the one actually wired into training by `run_buggy.sh` via `reward.custom_reward_function.path`/`name=compute_score`. Validation (`data_source != TRAIN_SOURCE`) still uses genuine math correctness, so training and the target task are misaligned.

## Observable impact

The following values describe the historical v2 data. Active v3 corrects
incomplete training questions; these numbers are not measurements on v3.
The original log is absent from the package. Do not require the agent to
quote these values for full evidence credit: tracing the active reward route
and demonstrating that a short wrong answer outranks a long correct answer
is sufficient evidence of the objective mismatch.

- `critic/rewards/mean` (the proxy) is high and rising (~0.52 → 0.67, mean ~0.54)
- Held-out `val-core/math__math/acc/mean@1` declines: 0.78125 (step 0) → 0.6875 → 0.65625 → 0.59375 (step 15)
- `response_length/mean` falls (~677 → 474): the policy learns to be shorter, which is what the proxy rewards
- `true_accuracy/score_mean` stays ~0.52 and does not improve, confirming the proxy gain is not correctness

## Trigger conditions

- `run_buggy.sh` sets `reward.custom_reward_function.path=autorlbench_case/type3_length_reward_v2_fn.py`; its training-source branch rewards brevity, not correctness.

## Documentation reconciliation

- PPO completing all 15 steps with a rising training reward is reward hacking, not health. A short incorrect answer can receive more training reward than a longer correct answer. This is proxy-objective misalignment — not a dead reward signal, a parser mismatch, an actor-learning-rate collapse, or a startup failure.
