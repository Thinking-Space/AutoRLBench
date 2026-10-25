# AIME25 single-sample variance

Evaluating a Qwen3-4B-Base PPO checkpoint on AIME25. One sample per question makes the score too noisy to judge the run, so a low score is a false alarm, not a training failure.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | configuration |
| **Tags** | `verl` `ppo` `sampling` `aime25` |
| **Model** | Qwen3-4B-Base (PPO-Critic checkpoint) |
| **Dataset** | AIME 2025 (30 questions) |
| **Framework** | verl, evaluated through `run_eval.py` |
| **Environment** | `ppo-critic-vimpo` · no GPU needed for grading |
| **Evaluations** | issue · solution |
| **Recorded runs** | the same checkpoint evaluated at 1 and at 32 samples per question |

## Difficulty

`config/eval_config.yaml` sets `evaluation.samples_per_question: 1` with `temperature: 1.0` and `top_p: 1.0` on a 30-question benchmark. One question is worth 3.3 points. Repeated evaluation of the *same* checkpoint gives per-epoch accuracy from 6.67% to 20%, and 10% shows up in 7 of 32 epochs.

The code and weights are fine. The hard part is to stop treating one low score as evidence of a PPO, reward, or optimizer bug, and to recognize that the measurement protocol itself is the defect. Controlled 1-sample vs 32-sample evidence ships in `workspace/autorlbench_case/original_eval/`.

## Reference solution

- **issue**: name `samples_per_question: 1` as the root cause. Training-failure explanations fail.
- **solution**: set `samples_per_question: 32` with the `mean` reducer and keep stochastic decoding. `run_eval.py --protocol-only` then reports `expected_generations: 960`.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: the agent repairs the config and submits `report.md` without running anything; a missing report scores 0. This task ships no `run_spec.json` — a protocol repair has no experiment to rerun — so grading is a deterministic, offline protocol check with no tolerance to calibrate. It runs `run_eval.py --protocol-only` and asserts `aime25`, 30 questions, 32 samples, `mean`, `temperature 1.0`, `top_p 1.0`, and 960 generations. It then runs `sha256sum -c` against the oracle manifest to confirm the preserved evidence was not modified.
