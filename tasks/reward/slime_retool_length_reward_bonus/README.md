# Length bonus reward hack

Training Qwen3-4B with GRPO on DAPO-Math-17k. The reward adds an unconditional length bonus, so the policy learns to write longer answers instead of correct ones.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | reward |
| **Tags** | `slime` `grpo` `reward-hacking` `length-bonus` `aime` |
| **Model** | Qwen3-4B (Instruct-2507, SFT → RL) |
| **Dataset** | DAPO-Math-17k (RL), ReTool-SFT (SFT warm-up), AIME-2024 (eval) |
| **Framework** | slime (Megatron + SGLang), `examples/retool/retool_qwen3_4b_rl.sh` |
| **Environment** | `slime-v0.3.2` · 8 GPUs |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective run (100 steps) · 1 repaired reference run (70 rollouts) |

## Difficulty

`examples/retool/generate_with_retool.py::reward_func` adds `5 * min(response_length / rollout_max_response_len, 1.0)` to every sample, on top of the exact-match score. The term ignores correctness, so a long wrong answer can outscore a short correct one.

Nothing fails loudly. The logged `eval/aime` is the mean of `reward_func` and includes the bonus, so it *rises*, from +0.82 to +3.84, above the healthy shaped-reward maximum of `1`. Meanwhile eval response length grows from 4.1k to 15.3k tokens, truncation from 0.8% to 82.5%, and the decontaminated score falls from about -0.417 to -0.831 (exact accuracy was not recorded). Unusual but correct recipe settings (`kl-loss-coef 0.00`, `rotary-base 5000000`, `max_turns=16`) are distractors.

## Reference solution

- **issue**: name the unconditional length term in `reward_func` as the root cause, and connect the out-of-range `eval/aime` to increasing length/truncation and the deteriorating decontaminated score. The reconstructed accuracy bounds are not measured accuracies. Calling the run healthy, or blaming a recipe setting, fails.
- **solution**: delete the length term. After that, `generate_with_retool.py` is byte-identical to the healthy recipe.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`; the weighted total alone is not sufficient.
- **solution**: the agent repairs the workspace and submits `report.md` without training; a missing report scores 0. The harness then reruns `examples/retool/retool_qwen3_4b_rl.sh` per `run_spec.json` with `--num-rollout 70 --eval-interval 10`, and grades only that run. The best `eval/aime` (correctness plus tool-call shaping) must fall in `[-0.55, 0.5]`. With the bonus still present, every eval point exceeds the upper bound. The repaired reference run peaks at -0.4246 (step 59, 0.125 margin), and step 0 (-0.5808) fails the lower bound, so a step-0-only submission fails. The bounds come from one run per side.

The grader requires validation at the declared final step and rejects evaluation
points outside the declared horizon before applying the existing best-score gate.
A passing earlier point alone does not establish completion.
