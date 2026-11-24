# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the recorded
workspace and the control run that fixed it.

Expected verdict: **unhealthy**.

## Root cause

- `data/pool_clean.jsonl` (part of the DAPO rows) + `scripts/make_rl_data.py`:
  `"ground_truth": str(r.answer).strip()`
  - kind: `data` / `script`
  - observed: 4,679 of the 11,579 DAPO-Math-17k rows (`"src":"dapo"`, about 40%;
    no field marks which) store the answer with a trailing U+200B ZERO WIDTH SPACE
    (`"answer":"34\u200b"`, bytes `34 e2 80 8b`), written raw rather than as a `\u200b`
    escape, so it renders as `"34"`. The other DAPO rows and all Skywork-OR1 /
    POLARIS answers are clean. `make_rl_data.py` reads the pool with
    `dtype={"answer": str}` and writes `str(r.answer).strip()`; `str.strip()`
    removes whitespace only, and U+200B is a format character (Unicode category
    `Cf`), so the affected ground truths in `data/rl_train.parquet` keep it
    (11.1% of the training set, 40% of its DAPO rows).
  - expected: ground truths without invisible characters, e.g. dropping category
    `Cf` characters (or keeping only the numeric characters) in
    `make_rl_data.py`, or normalizing the target the same way in
    `scripts/math_reward.py` before matching.

The reward side is where the invisible character becomes a zero:
`scripts/math_reward.py::compute_score` calls
`inspect_ai.scorer._common.match_str(answer, ground_truth, location="end", numeric=True)`.
The numeric comparison is used only when the target passes `str.isnumeric()`;
`"34\u200b".isnumeric()` is False, so it falls back to a string comparison, and
`"34"` never equals `"34\u200b"`.

A passing diagnosis must identify that a subset of the DAPO prompts scores
exactly zero because their ground truths carry an invisible zero-width character
from the pool (which `.strip()` does not remove), **and** connect this to the
numeric matcher returning False for those targets. Noting only that DAPO scores
lower, or that some correct answers score 0, without finding the extra character,
or attributing the gap to difficulty, formatting, or whitespace is partial or
wrong.

## Observable impact

- Primary source metric: `critic/score/mean` (training reward; no in-run
  validation), split by data source and by prompt. The rollout dumps carry no
  source field; the source comes from joining each prompt with `extra_info.src`
  in the parquet that `make_rl_data.py` builds, or with `src` in
  `data/pool_clean.jsonl`.
- Scored evidence (these three items are the evidence the judge should count):
  1. Inside DAPO, 20 prompts (160 samples in the five rollout dumps) score
     exactly 0 on all 8 samples, while the other 27 DAPO prompts score 0.593.
  2. Their `gts` values end with U+200B (visible in the bytes, `repr`, `len`, or
     `int()` failing), and so do the matching pool answers; no Skywork-OR1 or
     POLARIS `gts` does.
  3. Many of those zero-scored responses are actually correct once the character
     is removed (108 of 160 under `scripts/math_reward.py`; e.g. `ANSWER: 13`
     next to a `gts` that displays as `13`).

### Context (not scored)

- From `workspace/outputs/grpo/train.log`, `metrics.jsonl` and
  `rollouts/{1..5}.jsonl`: `critic/score/mean` 0.453 / 0.480 / 0.457 / 0.441 /
  0.355; `actor/grad_norm` 0.08-0.13; `response_length/clip_ratio` 0.39-0.50; by
  source DAPO 0.340 (376 samples), Skywork-OR1 0.546 (672), POLARIS 0.280 (232).
- Control reference (author-side; not shipped in the workspace and not expected
  in the diagnosis): same image, seed, data selection and budget, format
  characters dropped in `make_rl_data.py`: `critic/score/mean` 0.512 / 0.500 /
  0.551 / 0.477 / 0.574; by source DAPO 0.641, Skywork-OR1 0.533, POLARIS 0.302;
  clip ratio 0.37-0.50.

## Trigger conditions

- Part of a source whose answer strings carry an invisible format character
  (here a trailing U+200B, typical of text copied from web pages), cleaned only
  with `str.strip()`, and a reward that compares the target as a string or
  through a matcher whose numeric path requires `str.isnumeric()`.

## Wrong attributions (must not pass)

- "DAPO problems are harder" / "DAPO is the weak source": DAPO's 0.340 is the
  mixture of 27 prompts at 0.593 and 20 prompts at exactly 0; in the control DAPO
  is the easiest source (0.641), and 108 of the zero-scored responses are
  correct.
- "POLARIS is the weak source": POLARIS scores ~0.28-0.30 in both arms; it is
  genuinely harder and unaffected.
- Uniform-zero groups as ordinary hard prompts: these groups score 0 even though
  most of their samples end with the right answer.
- Answer extraction or format (missing `ANSWER:` line, `\boxed{}`): the affected
  responses end with a correct `ANSWER: N` line.
- Truncation / response length cap: 39-50% clipped, as in the control.
- Whitespace in the ground truth: `.strip()` is already applied; the character
  is not whitespace.
- A globally broken scorer or ground-truth set, model capacity, learning rate or
  KL settings: Skywork-OR1 scores ~0.55, clean DAPO prompts ~0.59, and the
  gradient is non-zero.
- Float ground truths (`"13.0"`): no `gts` in the rollout dumps has a decimal part.
- The tqdm / wandb shutdown tracebacks near the end of `train.log`: they come
  after the 5/5 progress bar (the step-5 metrics line is flushed after them) and
  do not affect training.

## Documentation reconciliation

- The lower `critic/score/mean`, DAPO's lower mean, and the uniform-zero GRPO
  groups of the affected prompts are symptoms of the corrupted ground truths, not
  separate causes.
- `scripts/math_reward.py` is the evaluation-style scorer and behaves as designed
  for clean integer-string targets; the defect is the value it receives.
