# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `autorlbench_case/buggy_reward_fn.py::compute_score`
  - kind: `symbol`
  - observed: answer extracted with `re.search(r"<answer>\s*(.*?)\s*</answer>", solution_str, re.DOTALL)`; on no match returns `{"score": 0.0, "acc": False, "parser_matched": False}`
  - expected: extract the boxed answer (`\boxed{...}`, last match) and grade it, so correct boxed answers receive reward
  - note: this reward is the one actually wired into training by `run_buggy.sh` via `reward.custom_reward_function.path`/`name=compute_score`. GuruMath prompts and ground truth use boxed answers, which the XML-tag regex never matches.

## Observable impact

- `critic/rewards/mean` and `critic/score/mean` are `0.0` at every step
- `val-aux/math__math/parser_matched/mean@1` is `0.0` (no response matched)
- `val-core/math__math/acc/mean@1` and `val-core/math__aime_repeated_8x/acc/mean@1` stay `0.0`
- `timing_s/reward` is non-zero, so the reward function runs but never matches

## Trigger conditions

- `run_buggy.sh` sets `reward.custom_reward_function.path=autorlbench_case/buggy_reward_fn.py`; the function's `<answer>...</answer>` parser cannot match boxed answers.

## Documentation reconciliation

- PPO continuing to execute normally (finite loss, healthy-looking gradients and MFU) is a symptom of the silent zero reward, not evidence of health. This is a reward-parser contract mismatch, not an optimizer, policy-collapse, or startup failure.
