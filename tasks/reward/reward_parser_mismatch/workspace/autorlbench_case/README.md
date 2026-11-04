# autorlbench_case

Scoring and staged data for the GuruMath smoke runs.

## Scoring modules

- `gurumath_qa_em.py` — the default scorer. Boxed-answer extraction
  (`\boxed{}`) plus exact-match grading through
  `verl.utils.reward_score.naive_dapo.grade_answer`. Wired as the default
  through `recipe/vimpo/config/vimpo_trainer.yaml`
  (`reward.custom_reward_function`).
- `math_reward.py` — the search_r1-style QA-contract variant
  (`<answer>...</answer>` extraction; see
  `verl/utils/reward_score/search_r1_like_qa_em.py`). Kept for the QA-tag data
  lane; reusing `grade_answer` keeps the two scorers comparable.
- `gsm8k_exact_match.py` — GSM8K-format grader (`#### <answer>`) for the
  GSM8K subset of the offline pipeline; not part of the GuruMath runs.

## Data

`assets/` holds the staged parquets (train / AIME / MATH-500) produced by the
offline QA pipeline; see the individual scorers above for the answer-format
contract each lane expects.
