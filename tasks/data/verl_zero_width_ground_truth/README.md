# An invisible character in one source's answers silently zeroes its GRPO rewards

Training Qwen/Qwen3-4B with verl GRPO on an integer-answer math pool built from Skywork-OR1, POLARIS-53K and DAPO-Math-17k: 40% of the DAPO answers end with an invisible U+200B ZERO WIDTH SPACE, the data script's `.strip()` keeps it, and the evaluation-style numeric matcher scores every sample of those prompts 0, even when the response ends with the right number. Nothing errors, the corrupted ground truths look like plain integers in every log, and the run looks like ordinary training with one somewhat harder source: mean reward ~0.44 instead of ~0.52.

| | |
| --- | --- |
| **Author** | Sikuan Yan |
| **Affiliation** | LMU Munich |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `unicode` `mixed-source-data` `partial-reward-loss` `ground-truth-corruption` |
| **Framework** | verl v0.8.0 tag (commit `7aed6b2`, shipped in the workspace; its version file reads 0.8.0.dev), GRPO; inspect_ai 0.3.150 installed by `prepare.sh` |
| **Run time** | ~34 minutes (5 steps) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

4,679 of the 11,579 DAPO-Math-17k rows of `data/pool_clean.jsonl` (40%; no field marks which) store their answer with a trailing U+200B (`"answer":"34\u200b"`, bytes `34 e2 80 8b`), written raw rather than as a `\u200b` escape, as text copied from web pages often is; the other DAPO rows and all Skywork-OR1 and POLARIS answers are clean. `scripts/make_rl_data.py` reads the pool with `dtype={"answer": str}` and writes `"ground_truth": str(r.answer).strip()`, which looks like the cleaning step it needs to be, but `str.strip()` removes whitespace only and U+200B is a format character (category `Cf`). `scripts/math_reward.py` mirrors the evaluator with `inspect_ai` `match_str(..., numeric=True)`, whose numeric path requires `target.isnumeric()`; `"34\u200b"` fails it and the string fallback never equals `"34"`. verl writes the rollout dumps with `ensure_ascii=False`, so a failing record shows a response ending `ANSWER: 13`, `"gts": "13"`, and score 0. 11.1% of the training set is affected; the gradient is non-zero and the aggregate reward is plausible for a 4B model on competition math.

The work is noticing that the reward is lower than it should be, localizing the loss below the source level, and explaining why visibly identical answers do not match. The dumps carry no source field, so splitting rewards by source means joining each prompt with `extra_info.src` in the parquet (or `src` in the pool). Split by source, DAPO scores 0.340 and looks like an ordinary harder source next to POLARIS (0.280); only split by prompt does it separate into 27 prompts at 0.593 and 20 prompts at exactly 0, whose zero-scored groups contain mostly correct answers. Inspecting a failing record by eye shows nothing wrong: the extra character only appears in the bytes, in `repr()`, in `len()`, or when `int()` rejects the string. Readings that fit the evidence and are wrong: DAPO or POLARIS is simply harder, the uniform-zero groups are ordinary hard prompts, answer extraction or format (the failing responses end with a correct `ANSWER: N` line), truncation (39–50% clipped, as in the control), and whitespace (`.strip()` is already applied).

| step | recorded `critic/score/mean` | control `critic/score/mean` | recorded `actor/grad_norm` | control `actor/grad_norm` | clip ratio recorded / control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.453 | 0.512 | 0.121 | 0.125 | 0.50 / 0.46 |
| 2 | 0.480 | 0.500 | 0.116 | 0.140 | 0.44 / 0.46 |
| 3 | 0.457 | 0.551 | 0.126 | 0.119 | 0.45 / 0.45 |
| 4 | 0.441 | 0.477 | 0.076 | 0.186 | 0.45 / 0.50 |
| 5 | 0.355 | 0.574 | 0.099 | 0.104 | 0.39 / 0.37 |

| subset (all 5 rollout dumps) | prompts | samples | recorded mean score | control mean score |
| --- | ---: | ---: | ---: | ---: |
| Skywork-OR1 | 84 | 672 | 0.546 | 0.533 |
| POLARIS | 29 | 232 | 0.280 | 0.302 |
| DAPO-Math-17k, all | 47 | 376 | 0.340 | 0.641 |
| DAPO, `gts` with U+200B | 20 | 160 | **0.000** | – |
| DAPO, clean `gts` | 27 | 216 | 0.593 | – |

108 of the 160 zero-scored DAPO responses score 1 under the same scorer once the character is removed from the target.

Horizon: 5 GRPO steps, no in-run validation (`trainer.test_freq=-1`); the evidence is the per-step training metrics (`workspace/outputs/grpo/train.log`, `metrics.jsonl`), all five rollout dumps, the pool and the two scripts, and the control arm. Both arms ran once (seed 1) in the shared `verl-vllm0.17.0` image on 2 × H100, start model `Qwen/Qwen3-4B` (revision `1cfa9a7`, pinned in `prepare.sh`), 32 prompts × 8 samples, response cap 8192, lr 1e-6, 5.5–6.8 minutes per step; the control differs only in dropping category-`Cf` characters before `.strip()` in `scripts/make_rl_data.py`, which restores the affected DAPO prompts and leaves everything else unchanged. With one run per arm the step-to-step noise is about ±0.05, so the aggregate gap (mean 0.437 vs 0.523) is suggestive on its own; the per-prompt evidence is what establishes the defect. The W&B `run-*.wandb` history file was removed because it embeds host paths and user/host names; `metrics.jsonl` carries the same per-step history. The tqdm / wandb shutdown tracebacks near the end of `train.log` come after the 5/5 progress bar (the step-5 metrics line is flushed after them) and do not affect training. The defect family was first met in an automated RL experiment, where integer answers turned into float strings (`"17.0"`) zeroed every reward; this version keeps the mechanism (a ground-truth string the numeric matcher rejects) but confines it to part of one source and makes the corruption invisible in the logs.

Construction note: the U+200B was injected by the author into 40% of the DAPO answers (rows chosen by an MD5 hash of the problem text) to reproduce, in a subset and invisibly, the real failure described above; everything else in the pool is as recorded.

Data: `data/pool_clean.jsonl` (81,882 problems) is the post-decontamination pool: integer-answer problems from Skywork-OR1 (math split, 55,366), POLARIS-53K (14,937) and DAPO-Math-17k (11,579), deduplicated and decontaminated against AIME 2025/2026; its assembly script is not included. 93 large answers are in lossy scientific notation (e.g. `2.040816327e+40`); those few ground truths are wrong under any normalization but are not why the affected prompts score zero. Licences: POLARIS-53K and DAPO-Math-17k are Apache-2.0 on Hugging Face; the Skywork-OR1-RL-Data card states no licence, and its GitHub repository (SkyworkAI/Skywork-OR1) is Apache-2.0.

Issue-only: each arm ran once, the aggregate margin is within two steps' noise, and repairs at different places are equally valid (clean the target in `make_rl_data.py` or in `math_reward.py`); the diagnosis is the gradeable part.

## Reference solution

- **issue**: identify that 20 of the 47 DAPO prompts in the dumps score exactly 0 because their ground truths (40% of the DAPO rows in the pool, 11.1% of the training set) end with an invisible U+200B that `.strip()` in `make_rl_data.py` does not remove, so `match_str(numeric=True)` falls back to a string comparison that never matches, even for correct answers. A repair drops format characters (or keeps only the numeric characters) when building or scoring the target. Attributions to DAPO or POLARIS difficulty, ordinary hard prompts, answer extraction or format, truncation, whitespace, float ground truths, model capacity, learning rate / KL settings, or a globally broken scorer do not pass; noting that DAPO scores lower, or that some correct answers score 0, without finding the extra character is partial.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. As implemented in the shared `judge.py`, passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
