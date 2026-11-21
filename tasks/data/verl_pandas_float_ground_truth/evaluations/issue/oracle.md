# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the recorded
workspace and the control run that fixed it.

Expected verdict: **unhealthy**.

## Root cause

- `data/pool_clean.jsonl` + `scripts/make_rl_data.py`: `"ground_truth": str(r.answer)`
  - kind: `data` / `script`
  - observed: every `answer` in `data/pool_clean.jsonl` is a JSON **float**
    (`"answer":15625.0`, `"answer":32.0`). The pool was assembled with string answers; the floats match
    one more pandas round trip during deduplication and decontamination:
    `read_json` infers dtypes, one answer is about 3.8e65 (beyond int64), so the
    whole column becomes `float64`, and `to_json` writes floats. `make_rl_data.py` reads
    the pool with `pd.read_json(..., lines=True)` and `str(r.answer)` writes every
    ground truth with a trailing `.0` (`"48.0"`, `"714.0"`, `"15625.0"`) into
    `data/rl_train.parquet`.
  - expected: integer ground truths (`"48"`, `"714"`), e.g.
    `str(int(float(r.answer)))` in `make_rl_data.py`, or stripping a trailing `.0`
    before matching in `scripts/math_reward.py`. Passing `dtype={"answer": str}` to
    `read_json` is **not** sufficient: the file already holds floats and the strings
    come back as `'15625.0'`.

The reward side is where the corrupted value becomes a zero:
`scripts/math_reward.py::compute_score` calls
`inspect_ai.scorer._common.match_str(answer, ground_truth, location="end", numeric=True)`.
With `numeric=True`, `match_str` compares numerically only when the target passes
`str.isnumeric()`; `"17.0".isnumeric()` is False, so it falls back to a string
comparison that never matches an extracted `17`
(`match_str("x 17", "17.0", ...) -> False`, `match_str("x 17", "17", ...) -> True`).

A passing diagnosis must identify that the ground truths carry a trailing `.0`
because the answers are floats (located in the pool data and/or the
`str(r.answer)` conversion in `make_rl_data.py`; naming pandas dtype inference as
how the floats arose is welcome but not required) **and** connect it to the numeric
matcher returning 0. A repair at either location (normalize in `make_rl_data.py`, or
in the reward) is correct. Saying only that the reward function "fails to match", or
proposing to normalize `.0` inside the reward without identifying where the `.0`
comes from, is a partial root cause. Attributing it to a dtype problem that
`dtype={"answer": str}` alone would fix is also partial, since that change does not
remove the `.0`.

Secondary, not the cause of the zero reward: 166 large answers in the pool are
already in lossy scientific notation (e.g. `2.040816327e+40`), so those few ground
truths are wrong under any normalization.

## Observable impact

- Primary source metric: `critic/score/mean` (training reward; the recorded run
  has no validation).
- From `workspace/outputs/grpo/train.log` and `workspace/math/grpo.jsonl`
  (5 steps): `critic/score/mean` = 0.0 at every step; `actor/grad_norm` = 0.0 and
  `actor/ppo_kl` = 0.0 exactly at every step (every GRPO group has identical
  rewards, so every advantage is 0).
- Responses are not the problem: mean length 5.9k-6.6k tokens,
  `response_length/clip_ratio` 0.41-0.48, identical to the control. In
  `workspace/outputs/grpo/rollouts/1.jsonl` all 256 `gts` values end in `.0`, 150 of
  256 responses close their reasoning, and 141 of 256 end with a number numerically
  equal to the ground truth; all score 0.
- Control run (same image, seed, data, budget; ground truths normalized to integers): reward
  0.516 / 0.504 / 0.559 / 0.496 / 0.551 at steps 1-5, `grad_norm` 0.11-0.13,
  clip ratio 0.40-0.47 (same as the recorded run).

## Trigger conditions

- A JSONL pool whose integer answers are stored as floats, here because a pandas
  `read_json`/`to_json` round trip inferred `float64` (one value outside int64 is
  enough; with all values in range the column would be int64 and stay integral).
- Ground truths built with `str()` on those floats, and a reward that compares the
  ground truth as a string or with a matcher whose numeric path requires
  `str.isnumeric()` targets.

## Wrong attributions (must not pass)

- Truncation / response length cap ("no final answer because responses are cut"):
  41-48% of responses are clipped, the same as in the healthy control (which scores
  ~0.5), yet the reward is exactly 0.0.
- Model too weak / data too hard: 141 of 256 step-1 answers are numerically right.
- Answer format (missing `ANSWER:` line, `\boxed{}`): the scorer takes the last
  number after `</think>` regardless; `fmt` is ~0.48 in both arms.
- Reward function not called / crashing: it returns a valid 0.0.
- Learning rate too small, KL/clip settings, or "training did not improve": the
  gradient is exactly zero because all advantages are zero.
- Uniform-reward GRPO groups as a normal sparse-reward effect: here every group is
  uniform at 0.

## Documentation reconciliation

- `grad_norm = 0`, `ppo_kl = 0`, `pg_loss = 0` are symptoms of zero advantages,
  not separate causes.
- `scripts/math_reward.py` is the evaluation-style scorer and behaves as designed
  for integer-string targets; the defect is the value it receives.
