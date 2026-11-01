# Canonical solution oracle

## Root cause

- `autorlbench_case/buggy_reward_fn.py::compute_score`
  - kind: `symbol`
  - observed: answer extracted with `re.search(r"<answer>\s*(.*?)\s*</answer>", solution_str, re.DOTALL)` — never matches the boxed GuruMath answers, so every sample scores `0.0`
  - expected: extract the boxed answer and grade it against ground truth
  - note: this is the reward actually wired into training by `run_buggy.sh`
    (`reward.custom_reward_function.path`/`name=compute_score`).

## Observable impact

- Primary source evaluation metric: `val-core/math__math/acc/mean@1`.
- Reward and validation accuracy are flat at `0.0` because `grade_answer` is
  never reached; PPO advantages are noise around zero.

## Trigger conditions

- `autorlbench_case/buggy_reward_fn.py` uses the `<answer>...</answer>` parser while
  prompts and ground truth are boxed (`\boxed{...}`).

## Required remediation

- Replace the XML-tag extraction with boxed extraction (take the last
  `\boxed{...}` expression with balanced nested braces) and grade it, keeping the
  `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  signature and the `{"score", "acc", "parser_matched"}` return contract so it
  stays compatible with the PPO custom-reward interface. A valid reference
  implementation is `fixed_reward_fn.py` in this directory. With ground truth
  `16`, `\boxed{16}` must score `1` and `\boxed{17}` must score `0`.
  (`grade_answer` comes from `verl.utils.reward_score.naive_dapo`, unchanged.)

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: the terminal (final-step) `val-core/math__math/acc/mean@1` must be
  at least `required_final_score` (see `grader_spec.json`); the terminal value is
  used, not a peak. The historical final baseline is `0.0`.
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `required_training_reward`.
- final `score = 0.8 * validation + 0.2 * training_reward`.

Pass requires both components (`score == 1.0`). The framework still records the
weighted float when only one component passes.

The reference extraction reuses `_last_boxed_only_string` from the bundled
math scorer. Correct fractions and radicals such as `\boxed{\frac{17}{8}}`
and `\boxed{\sqrt{2}}` must reach `grade_answer`; a flat-brace regex is insufficient.
