# Canonical solution oracle

## Root cause

- `calc_agent/calc_agent_loop.py::CalcAgentLoop.run`
  - kind: `symbol`
  - observed: the training sample is `apply_chat_template(agent_data.messages, tools=...,
    add_generation_prompt=False)` minus the prompt rendering, with a mask from
    `_assistant_token_mask` and the rollout log-probs carried over to the assistant positions in
    order. The transcript has no reasoning, so the Qwen3 template renders earlier turns without
    `<think>` and the last with an empty `<think></think>`.
  - expected: the sampled token stream, `agent_data.prompt_ids` split at
    `len(agent_data.response_mask)`, `agent_data.response_mask`, `agent_data.response_logprobs`.

## Observable impact

- Primary source evaluation metric: `training/rollout_actor_probs_pearson_corr` (verl's
  correlation between the rollout engine's token probabilities and the trainer's recomputed ones
  on response tokens). Defective -0.02..0.04 every step, control 0.986..0.995.
- Held-out `val-core/openai/gsm8k/acc/mean@1` and `val-core/reasoning-machines/gsm-hard/acc/mean@1`
  do not separate the arms (see `recorded_comparison/COMPARISON.md`).

## Trigger conditions

- Same as `evaluations/issue/oracle.md`: thinking mode, an OpenAI-style transcript without
  reasoning, and a training sequence rebuilt from it.

## Required remediation

- `calc_agent/calc_agent_loop.py`: restore the token-stream block of `ToolAgentLoop.run`
  (`code_target/calc_agent/calc_agent_loop.py`, `fix.diff`). Removing `_assistant_token_mask`
  and the two chat-template imports is cleanup, not required.

Machine grading reads only the new experiment logs.
`training/rollout_actor_probs_pearson_corr` at `final_step` 100 (`score_mode: final`) must be
≥ 0.9 and ≤ 1.0. Pass is score 1, anything else 0.
