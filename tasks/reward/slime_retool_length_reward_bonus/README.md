# Length bonus reward hack

Training Qwen3-4B with GRPO on DAPO-Math-17k. Two independent defects were injected: the reward adds an unconditional length bonus, and the tool-call parser reads the wrong JSON key, so correctly-formatted code-interpreter calls never execute. The policy learns to write longer answers instead of correct ones.

| | |
| --- | --- |
| **Author** | Yinghao Chen |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | `reward-hacking` `length-bonus` `tool-call-parsing` |
| **Framework** | slime (GRPO), Megatron + SGLang |
| **Run time** | ~3.7 hours |
| **Resources** | 8 GPU |

## Difficulty

`examples/retool/generate_with_retool.py` carries both defects. `reward_func` adds `5 * min(response_length / rollout_max_response_len, 1.0)` to every sample, on top of the exact-match score — the term ignores correctness, so a long wrong answer can outscore a short correct one. The tool-call parser reads `tool_call_data.get("parameters", {})` while the chat template `TOOL_TEMPLATE` instructs `{"name": ..., "arguments": ...}`, so template-conforming calls carry empty arguments and the code interpreter never runs the requested code.

Nothing fails loudly. The logged `eval/aime` is the mean of `reward_func` and includes the bonus, so it *rises*, from +0.356 to +3.014, outside the healthy shaped-reward range `[-1, 1]`. Meanwhile eval response length grows from about 3.9k to 12.4k tokens, truncation from about 0.8% to 47.3%, and the decontaminated held-out score stays at about -0.77, bounding exact-match accuracy at about 11.6% at step 69. Unusual but correct recipe settings (`kl-loss-coef 0.00`, `rotary-base 5000000`, `max_turns=16`) are distractors.

## Reference solution

- **issue**: name both independent defects — the unconditional length term in `reward_func` and the `parameters`/`arguments` key mismatch in the tool-call parser — with their correct causal roles; naming only one is partial. Connect the out-of-range `eval/aime` to increasing length/truncation and the flat decontaminated score. The reconstructed accuracy bounds are ceilings, not measured accuracies. Calling the run healthy, or blaming a recipe setting, fails.
- **solution**: delete the length term and repair the tool-call parser to read the key the chat template instructs. Fixing only the length term leaves `eval/aime` at about -0.845 at step 69, below the acceptance band.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`; the weighted total alone is not sufficient.
- **solution**: the agent repairs the workspace and submits `report.md` without training; a missing report scores 0. The harness then reruns `examples/retool/retool_qwen3_4b_rl.sh` per `run_spec.json` with `--num-rollout 70 --eval-interval 10`, and grades only that run. The final `eval/aime` (correctness plus tool-call shaping) at step 69 must fall in `[-0.55, 0.5]`. With the bonus still present, the final point exceeds the upper bound; the length-term-only repair sits at -0.845. The repaired reference run reaches -0.4861 at step 69 (peak -0.4246 at step 59), so a graded horizon that stops earlier fails. The bounds come from one run per side. The verifier hashes the read-only scoring assets (`math_dapo_utils.py`, `tool_sandbox.py`, the staged datasets) and pattern-checks the launch script's scoring and eval wiring after the rerun; touching either scores 0.

The grader requires validation at the declared final step and rejects evaluation
points outside the declared horizon before applying the final-score gate.
A passing earlier point alone does not establish completion.
