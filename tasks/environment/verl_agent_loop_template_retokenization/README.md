# Agent loop trains on a re-rendered transcript

Training Qwen3-1.7B (thinking mode) with GRPO on GSM8K with a calculator tool through a verl 0.9.0 agent loop. The loop trains on the chat-template rendering of the episode transcript instead of the tokens vLLM sampled, so the reasoning never reaches the loss and the run is silently off-policy.

| | |
| --- | --- |
| **Author** | Ziqi Zhou |
| **Domain** | environment |
| **Tags** | `verl` `grpo` `multi-turn` `tool-calling` `chat-template` `gsm8k` |
| **Model** | Qwen3-1.7B (thinking mode) |
| **Dataset** | GSM8K train 7,473 / held-out 512 GSM8K test + 512 GSM-Hard |
| **Framework** | verl v0.9.0 (`483b8a0`) agent loop, FSDP + vLLM 0.28 colocated, `run_calc_grpo.sh` |
| **Environment** | `verl-0.9-vllm0.28` (not yet built) · 1 × 96 GB GPU (recorded on RTX PRO 6000) |
| **Evaluations** | issue · solution |
| **Recorded runs** | 2 defective + 2 control runs (two pairs, one run each per pair), 100 steps, eval every 10. Defective ≈ 2 to 2.3 h, control ≈ 3.9 to 4.8 h per run |

## Difficulty

`calc_agent/calc_agent_loop.py::CalcAgentLoop.run` drives the stock token-in/token-out `ToolAgentLoop` state machine, then rebuilds the training sample from the episode transcript:

```python
prompt_ids = await self.apply_chat_template(raw_prompt, tools=self.tool_schemas)
full_ids = ... apply_chat_template(self.tokenizer, agent_data.messages, tools=..., add_generation_prompt=False)
response_ids = full_ids[len(prompt_ids):]
response_mask = self._assistant_token_mask(response_ids)
# Carry the rollout log-probs over to the assistant positions, in order.
```

The transcript holds OpenAI-style assistant messages (text after `</think>` plus tool calls). The Qwen3 template renders earlier assistant turns with no `<think>` block and the last one with an empty `<think>\n\n</think>`. Of 16,684 sampled response tokens in the first-step dumps only 4,417 reach the loss, and on those surviving tokens KL(rollout ‖ trainer) is 0.22 nats/token because their context changed.

Nothing fails. Shapes line up, the reward is read from the transcript and is correct, and `actor/ppo_kl` stays near 0 because the old log-probs are recomputed on the same wrong sequence. Held-out accuracy does **not** separate the arms at this budget (table below), so "the run is fine" and "accuracy collapsed" are both wrong readings. The strongest evidence is verl's own rollout-vs-trainer diagnostic: `training/rollout_actor_probs_pearson_corr` is -0.02 to 0.04 every step, against 0.986 to 0.995 for the control on the same stack. That rules out the tempting diagnosis of bf16 vLLM-vs-FSDP numerical mismatch (that gives about 0.99). Other wrong diagnoses: the calculator (the policy stops calling it, `num_turns/mean` 3.90 → 2.07, because tool-call turns are trained without the reasoning before them), the reward, the learning rate or KL coefficient, and "multi-turn RL is just unstable".

| Step | defective GSM8K | control GSM8K | defective GSM-Hard | control GSM-Hard | defective `rollout_corr/kl` | control `rollout_corr/kl` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 88.7% | 88.3% | 65.4% | 65.6% | n/a | n/a |
| 50 | 86.5% | 88.9% | 66.0% | 67.2% | 1.27 | 0.0013 |
| **100** | **89.1%** | **88.5%** | **67.8%** | **67.8%** | **2.99** | **0.0016** |

The replicate pair (GSM8K held-out only) shows the same picture: defective 85.9% → 86.5%, control 89.3% → 89.1%, with the same base weights already 3.4 points apart at step 0.

## Reference solution

- **issue**: name `CalcAgentLoop.run` rebuilding the training sequence with `apply_chat_template` over the reasoning-free transcript instead of the sampled token stream, and connect it to the near-zero rollout/actor correlation and the missing reasoning tokens. Blaming bf16/vLLM numerics (and prescribing TIS or rollout correction), the tool, the reward, the learning rate, or claiming an accuracy collapse does not pass.
- **solution**: return the sampled stream, as `ToolAgentLoop.run` does: `prompt_ids`/`response_ids` split from `agent_data.prompt_ids` at `len(agent_data.response_mask)`, with `agent_data.response_mask` and `agent_data.response_logprobs` (`evaluations/solution/oracle/fix.diff`). The transcript stays for the reward and logs. Keeping `reasoning_content` in the transcript is not enough: it removes most of the drift here, but re-tokenization still changes 2 of 8 first-step episodes, and the template still drops reasoning before any later user turn.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. A run passes only if the verdict score is 1.0 and the root-cause and evidence scores are both above 0.8. The weighted score is reported but does not decide the pass.
- **solution**: rerun 100 steps. `training/rollout_actor_probs_pearson_corr` at step 100 (`score_mode: final`) must be ≥ 0.9. Recorded values: defective runs 0.007 and -0.009 at step 100 (never above 0.044 at any step), control runs 0.988 and 0.989 (never below 0.986). The threshold is 0.86 above the defective maximum and 0.086 below the control minimum. It is a mismatch metric, not accuracy, because no accuracy threshold separates the arms. Calibrated on two runs per side, so it supports margins, not variance. Known limit: a "fix" that keeps the re-rendered sequence but fetches matching log-probs for it would pass this check while still training on the wrong tokens. The issue tier grades the attribution.
