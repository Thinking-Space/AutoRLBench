# pandas turns integer ground truths into floats, so every GRPO reward is zero

Training Qwen/Qwen3-4B with verl GRPO on an integer-answer math pool (Skywork-OR1, POLARIS-53K, DAPO-Math-17k): the pool file stores every integer answer as a float (an earlier pandas dtype-inference round trip), the data script turns it into ground truths like `"17.0"`, and the evaluation-style numeric matcher scores every sample 0, so the policy gradient is exactly zero from step 1.

| | |
| --- | --- |
| **Author** | Sikuan Yan |
| **Affiliation** | LMU Munich |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `silent-zero-reward` `dtype-inference` `ground-truth-corruption` `zero-advantage` |
| **Framework** | verl v0.8.0 tag (commit `7aed6b2`, shipped in the workspace; its version file reads 0.8.0.dev), GRPO; inspect_ai 0.3.150 installed by `prepare.sh` |
| **Run time** | ~33 minutes (5 steps) |
| **Resources** | 2 × H100 (80 GB) |

## Difficulty

`data/pool_clean.jsonl` stores every `answer` as a JSON float (`"answer":15625.0`). The pool was assembled from the three sources with string answers; the floats match one more pandas round trip during deduplication and decontamination: `read_json` infers dtypes, one answer is about 3.8e65 (beyond int64), so the whole column becomes `float64`, and `to_json` writes floats. In the image (pandas 3.0.1), a three-row file with the string answers `"15625"`, `"17"` and the 66-digit value round-trips to `15625.0`, `17.0`, `3.80142381e+65`. `scripts/make_rl_data.py` reads the pool with `pd.read_json(..., lines=True)` and writes `"ground_truth": str(r.answer)`, so the parquet holds `"48.0"`, `"714.0"`. Asking for `dtype={"answer": str}` at this point does not help: the file already holds floats and the strings come back as `'15625.0'`. `scripts/math_reward.py` mirrors the evaluator with `inspect_ai` `match_str(..., numeric=True)`, whose numeric path requires `target.isnumeric()`; `"17.0"` fails it and the string fallback never matches an extracted `17`. Nothing raises and nothing is logged: the run trains for 5 steps with a valid but constant reward. The symptom (every reward exactly 0) is obvious; the work is locating where the `.0` comes from and why the matcher rejects it. Separately, 166 large answers in the pool are already in lossy scientific notation (e.g. `2.040816327e+40`); those few ground truths are wrong under any normalization, but they are not why the reward is zero.

The obvious diagnoses are wrong. Truncation: 40–48% of responses are clipped (Qwen3-4B thinks long under the 8k cap), the same fraction as in the control, which scores ~0.5. Model capacity: 141 of the 256 step-1 responses end with the numerically correct answer. Learning rate: `grad_norm`, `pg_loss` and `ppo_kl` are exactly 0 because every advantage is 0. The decisive evidence is in the rollout dumps: all 256 `gts` end in `.0`.

| step | recorded `critic/score/mean` | control `critic/score/mean` | recorded `actor/grad_norm` | control `actor/grad_norm` | clip ratio recorded / control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000 | 0.516 | 0.0 | 0.124 | 0.48 / 0.46 |
| 2 | 0.000 | 0.504 | 0.0 | 0.134 | 0.45 / 0.44 |
| 3 | 0.000 | 0.559 | 0.0 | 0.122 | 0.41 / 0.41 |
| 4 | 0.000 | 0.496 | 0.0 | 0.106 | 0.48 / 0.47 |
| 5 | 0.000 | 0.551 | 0.0 | 0.125 | 0.42 / 0.40 |

Horizon: 5 GRPO steps, no in-run validation (`trainer.test_freq=-1`); the evidence is the per-step training metrics (`workspace/math/grpo.jsonl`, `workspace/outputs/grpo/train.log`), three rollout dumps, and the control arm. Both arms ran once (seed 1) in the shared `verl-vllm0.17.0` image on 2 × H100, start model `Qwen/Qwen3-4B` (revision `1cfa9a7`), 32 prompts × 8 samples, response cap 8192, lr 1e-6, ~6.3 minutes per step; the control differs only in normalizing the ground truths to integers (`str(int(float(answer)))` in `scripts/make_rl_data.py`, plus stripping a trailing `.0` in `scripts/math_reward.py`; either change alone restores the reward). The defect was first found in an automated RL experiment, where the zero scores of a 2-step smoke test were at first read as "expected given truncation" until the rollouts were dumped.

Data: `data/pool_clean.jsonl` (81,882 problems) is the post-decontamination pool: integer-answer problems from Skywork-OR1 (math split), POLARIS-53K and DAPO-Math-17k, deduplicated, decontaminated against AIME 2025/2026. Licences: POLARIS-53K and DAPO-Math-17k are Apache-2.0 on Hugging Face; the Skywork-OR1-RL-Data card states no licence, and its GitHub repository (SkyworkAI/Skywork-OR1) is Apache-2.0.

Issue-only: the control already scores ~0.52 at step 1, so no threshold above the control's step-0 value separates a repaired run within this budget, and each arm ran once.

## Reference solution

- **issue**: identify that the ground truths carry a trailing `.0` because the pool stores the answers as floats (visible in `data/pool_clean.jsonl`; `scripts/make_rl_data.py`'s `str(r.answer)` carries it into the parquet) and connect it to `match_str(numeric=True)` returning False for non-`isnumeric()` targets, hence all rewards 0 and zero advantages. A repair normalizes the ground truths to integers in `scripts/make_rl_data.py` or strips the `.0` before matching in `scripts/math_reward.py`; adding `dtype={"answer": str}` alone does not fix it. Attributions to truncation, model capacity, answer format, learning rate / KL settings, a crashing reward function, or "training did not improve" do not pass; blaming only the reward matcher without locating the `.0` source is partial.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. As implemented in the shared `judge.py`, passing requires a verdict score of 1.0 and root-cause and evidence scores above 0.8.
