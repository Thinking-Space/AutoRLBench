# Math-Verify scores zero inside reward workers

Training Qwen2.5-1.5B-Instruct with GRPO on a GSM8K + MATH mixture. Every MATH sample is scored 0 by a grader that fails silently in verl's reward-worker threads, while GSM8K accuracy and training reward rise as if the run were healthy.

| | |
| --- | --- |
| **Author** | Jianing Yin |
| **Domain** | reward |
| **Tags** | `verl` `grpo` `math-verify` `reward-loop` `threading` |
| **Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Dataset** | GSM8K main + MATH-lighteval, 7473 + 7500 train / GSM8K test 1319 + MATH-500 497 |
| **Framework** | verl @ `6f2aa24` with the pre-#5635 `math_verify.py`, GRPO, `run_grpo.sh` |
| **Environment** | `verl` · 2 GPUs · math-verify 0.9.0 installed into the workspace by `prepare.sh` |
| **Evaluations** | issue |
| **Recorded runs** | 2 defective + 2 repaired runs, 100 steps each (seed 1 on 2 GPUs, ~45 min; seed 2 on 1 GPU, ~70 min) |

## Difficulty

`reward/mixed_math_reward.py` scores GSM8K with verl's strict `####` scorer and every other source with `verl/utils/reward_score/math_verify.py::compute_score`. That function calls math-verify's `math_metric`, whose `parse()` installs a `SIGALRM` handler with `signal.signal()` for its timeout. verl's reward loop runs a synchronous `compute_score` through `loop.run_in_executor`, so the call happens in a worker thread, where `signal.signal()` raises `ValueError`. The function's bare `except Exception: pass` returns 0.0. The workspace is verl `6f2aa24` with this one file at its pre-fix revision `3d274f6` (fixed upstream by verl PR #5635).

Nothing fails and nothing is logged. GSM8K accuracy goes from 0.081 to 0.766 and training reward from 0.068 to 0.424, so "training does not work" is wrong, and calling the scorer from a Python shell scores the same answers correctly, which hides the cause. The strongest evidence is that MATH-500 logs exactly 0.0 already at step 0, before any update, while about 55% of those step-0 answers are correct by any offline grader.

| step | val GSM8K (logged) | val MATH-500 (logged) | val MATH-500 (offline re-grade) |
| ---: | ---: | ---: | ---: |
| 0 | 0.0811 | 0.0 | 0.5634 |
| 50 | 0.7415 | 0.0 | 0.5433 |
| 100 | 0.7657 | 0.0 | 0.5272 |

MATH-500 is evaluated on 497 of its 500 problems in every run: `data.filter_overlong_prompts=True` drops the three whose chat-templated prompt (Qwen2.5-1.5B-Instruct tokenizer) is longer than `data.max_prompt_length=512` tokens, namely `test/counting_and_probability/181.json` (711 tokens), `test/counting_and_probability/282.json` (744) and `test/prealgebra/1044.json` (835). The offline re-grade uses math-verify 0.9.0 `parse` and `verify` called directly in the main thread; details are in `evaluations/issue/reference/evidence.md`.

## Reference solution

- **issue**: name `compute_score` in `verl/utils/reward_score/math_verify.py` as the root cause: math-verify's signal-based timeout raises when the reward loop calls it off the main thread, and the broad `except` converts that into a zero MATH reward. Attributions to model capability, the `\boxed{}` format, truncation, a missing math-verify install, the reward router, or "training did not improve" do not pass.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle and scores
  verdict, root cause and evidence. As implemented in the shared `judge.py`, a run passes only if
  the verdict score is 1.0 and the root-cause and evidence scores are both strictly greater than 0.8;
  the 0.2 / 0.6 / 0.2 weighted score is reported but does not decide the pass.

This submission covers the issue level. Across two seeds the step-100 MATH-500 gap between the repaired and defective runs was +5.0 points (p=0.007) and +0.8 points (p=0.74), and the repaired run's own gain over its step 0 (+2.8 points in both) was not significant (p=0.17 and p=0.12), so a stable acceptance threshold based on MATH learning gain has not been established. The grading-path evidence (exception trace, cross-grading of the same answers, real reward paths on identical answers) is in `evaluations/issue/reference/`.
