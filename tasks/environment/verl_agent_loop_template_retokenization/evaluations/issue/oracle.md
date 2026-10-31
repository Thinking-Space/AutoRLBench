# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget, only `CalcAgentLoop.run` differs).

Expected verdict: **unhealthy**.

## Root cause

- `calc_agent/calc_agent_loop.py::CalcAgentLoop.run` (the block after the agent state machine,
  "Training sequence: render the finished transcript with the model's chat template")
  - kind: `symbol`
  - observed: after the episode, the loop throws away the token stream it exchanged with vLLM
    (`agent_data.prompt_ids` / `response_mask` / `response_logprobs`) and rebuilds the training
    sample by calling `apply_chat_template(self.tokenizer, agent_data.messages, tools=...,
    add_generation_prompt=False)` on the episode transcript. `response_mask` is recomputed by
    scanning the rendered ids for assistant blocks (`_assistant_token_mask`), and the rollout
    log-probs are carried over "to the assistant positions, in order".
    The transcript (`_transcript_message`) keeps only the text after `</think>` plus the tool
    calls, as an OpenAI-style message would. The Qwen3 chat template then renders every earlier
    assistant turn with no `<think>` block at all and the last assistant turn with an **empty**
    `<think>\n\n</think>\n\n`. The policy is updated on token sequences it never sampled:
    most of what it did sample (the reasoning) never reaches the loss, the tool-call and answer
    tokens that do reach it sit in a context the policy never saw, and every rollout log-prob is
    paired with a different token than the one it was measured on.
  - expected: return the token stream as sampled (token-in/token-out), exactly as
    `ToolAgentLoop.run` does:
    `prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]`,
    `response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]`,
    `response_mask = agent_data.response_mask`, `response_logprobs = agent_data.response_logprobs`.
    The transcript may still be used for the reward and for logging.

Naming this cause with the correct causal role (training tokens rebuilt from the chat template /
transcript instead of the sampled token ids, so training is off-policy and the logged
rollout-vs-trainer mismatch is real) is a complete attribution. Do not require additional causes.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1` (512 GSM8K test problems,
  temperature 0.6, one sample), with `val-core/reasoning-machines/gsm-hard/acc/mean@1` (512
  GSM-Hard problems) as the second held-out set. **Neither separates the defective run from the
  control** at this 100-step budget: the defective run's GSM8K accuracy moves 88.7% -> 86.5%
  (step 50) -> 89.1% (step 100) and GSM-Hard 65.4% -> 67.8%. The control and a second pair of
  runs land within the same 85-90% band (step-0 accuracy of the identical base weights already
  differs by up to 3.4 points between runs). The defect is visible in the rollout/trainer
  mismatch and in the policy's behaviour, not in held-out accuracy.
- verl's own rollout-vs-trainer diagnostics, logged every step in `results/run100`:
  `rollout_corr/kl` 0.67 at step 1 rising to 2.99 at step 100 (control 0.0012-0.0018),
  `training/rollout_actor_probs_pearson_corr` -0.02..0.04 (control 0.986-0.995),
  `training/rollout_probs_diff_mean` 0.16-0.22 (control ~0.005), `rollout_corr/ppl_ratio`
  6.4 at step 1 and up to 6,959 (control 1.001-1.002).
- `response_length/mean` is 86-177 tokens although the policy samples ~600-1,100 tokens per
  episode in thinking mode (the control logs 600-1,060 for the same recipe). The trainer only
  sees the tool calls and final answers.
- Episode dumps (`results/run100/episodes/episodes_sample.jsonl`): `stream_response_ids` (what
  vLLM sampled) and `response_ids` (what the loop returned) differ in every training episode. The
  decoded stream starts with `<think>\n...` while the returned response starts directly with
  `<tool_call>`, and its last assistant turn opens with an empty `<think>\n\n</think>`.
  `rollout_data/*.jsonl` shows the same short, reasoning-free outputs on the trainer side.
- Offline measurement on the 8 dumped episodes of the first training step (same base weights
  for vLLM and an HF bf16 recompute): of 16,684 sampled response tokens only 4,417 (26%) appear
  in the training sequence (per-episode mean: 85% dropped). 0.2% of the trained tokens were never
  sampled. On the surviving tokens, same token, rollout context vs. training context, 
  KL(rollout || trainer) is 0.22 nats/token and 19% of their probability ratios fall outside
  [0.8, 1.2]. Summed over an episode the log-ratio is about -124 nats. On the control the same
  measurement is 0.0014 nats/token, 1.7% outside [0.8, 1.2] (bf16 noise), and all 8 streams
  round-trip exactly.
- Behaviour drifts: `num_turns/mean` falls from 3.90 to 2.07 and tool calls per training episode
  fall from ~1 to ~0, the policy stops using the calculator, while the training score stays
  0.85-0.95.

## Trigger conditions

- Thinking mode with a reasoning-free transcript: the Qwen3 template inserts reasoning only from
  `reasoning_content` or from a `</think>` inside `content`, and renders earlier turns of a
  multi-step conversation without it. Any agent loop whose messages follow the OpenAI convention
  (reasoning dropped) loses all reasoning on re-render. For conversations with more than one user
  query the template drops earlier reasoning even when it is supplied.
- Re-tokenizing instead of reusing the engine's ids also changes whitespace and JSON
  serialisation of tool calls. Those are small next to the dropped reasoning here.
- The log-prob carry-over keeps the tensors the same shape, so verl raises nothing. GRPO, KL loss
  and the PPO ratio all run on the rebuilt sequence (`actor/ppo_kl` stays near 0 because
  `old_log_prob` is recomputed on the same wrong sequence).

## Documentation reconciliation

- **Held-out accuracy does not show the defect at this budget.** A diagnosis must not claim an
  accuracy collapse, and it must not call the run healthy because accuracy is flat. The run is
  unhealthy because the policy is optimised on sequences it never produced. The frozen mismatch
  metrics and episode dumps establish that directly.
- Reading the high `rollout_corr/kl` / low `rollout_actor_probs_pearson_corr` as numerical
  rollout/trainer mismatch (bf16 vLLM vs FSDP, kernel or precision differences) and prescribing
  truncated importance sampling or `rollout_correction` is a wrong diagnosis: numerical mismatch
  gives a Pearson correlation near 0.99, not near 0, and the control with the same stack logs
  0.0012-0.0018. Importance weights would only mask the symptom.
- The calculator tool is not the cause: it returns correct results, and the policy stops calling
  it because tool-call episodes are trained without the reasoning that preceded the call.
- The reward is not the cause: `gsm8k_score` reads the final assistant message, which is the
  same text in both arms, and train/held-out scores agree with the dumps.
- The learning rate (1e-6), KL loss (0.001, low_var_kl), `max_assistant_turns=6`,
  `response_length=4096`, the 1-GPU colocated setup, temperature 1.0 and the 100-step budget are
  shared recipe settings, not the cause. "Multi-turn RL is just unstable" is not a diagnosis.
- `trainer.use_v1=False` (legacy trainer, because `transfer_queue` is not installed), W&B offline
  mode and the hermes "Failed to decode tool call" warnings are environment noise.
- A diagnosis that names the re-rendered transcript as the training sequence, with the dropped
  `<think>` reasoning / chat-template rewrite as the mechanism, should receive full
  `root_cause_score`.
