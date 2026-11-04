# Canonical solution oracle

## Root cause

- `autorlbench_case/math_reward.py::compute_score`
  - kind: `symbol`
  - observed: answer extracted with `re.search(r"<answer>\s*(.*?)\s*</answer>", solution_str, re.DOTALL)` — never matches the boxed GuruMath answers, so every sample scores `0.0`
  - expected: extract the boxed answer and grade it against ground truth
  - note: this is the reward actually wired into training — `run_main.sh` sets
    `SMOKE_SCORING=1`, the recipe script sources `recipe/vimpo/config/scoring.env`
    under that flag, and its `REWARD_OVERRIDE` carries
    `reward.custom_reward_function.path`/`name=compute_score` onto the Hydra
    command line (superseding the `gurumath_qa_em.py` default in
    `vimpo_trainer.yaml`).

## Observable impact

- Primary source evaluation metric: `val-core/math__math/acc/mean@1`.
- Reward and validation accuracy are flat at `0.0` because `grade_answer` is
  never reached; PPO advantages are noise around zero.

## Trigger conditions

- `autorlbench_case/math_reward.py` uses the `<answer>...</answer>` parser while
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

Machine grading reads only the new experiment logs. Scoring is validation-first
(`score_mode: final` in `grader_spec.json`):

- validation: the terminal (final-step) `val-core/math__math/acc/mean@1` must be
  at least `required_final_score` (see `grader_spec.json`); the terminal value is
  used, not a peak. The historical final baseline is `0.0`. Whenever the fresh
  run logged any validation point, this gate alone decides the outcome.
- training reward: mean `critic/rewards/mean` over training steps strictly above
  `required_training_reward` is only a fallback gate for runs that produced no
  validation metric at all.
- final `score = 1.0` on pass and `0.0` otherwise.

The reference extraction reuses `_last_boxed_only_string` from the bundled
math scorer. Correct fractions and radicals such as `\boxed{\frac{17}{8}}`
and `\boxed{\sqrt{2}}` must reach `grade_answer`; a flat-brace regex is insufficient.
