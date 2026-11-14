# Search-token reconstruction drops the end token and collapses multimodal GRPO

Training Qwen/Qwen3-VL-4B-Thinking with GRPO on locaszzz/maestro_data (9200 training examples / 100 held-out examples), a first-turn search-token reconstruction defect corrupts tool-call delimiters and the recorded run collapses by step 200.

| | |
| --- | --- |
| **Author** | Jinyang Wu |
| **Domain** | algorithm |
| **Tags** | `maestro` `verl` `grpo` `multimodal` `tool-use` `token-truncation` |
| **Model** | Qwen/Qwen3-VL-4B-Thinking |
| **Dataset** | locaszzz/maestro_data |
| **Framework** | verl + verl_tool (Maestro), `run.py` → `scripts/train_4_gpus.sh` |
| **Environment** | `maestro-vllm011` · 8 GPUs: 4 tool-service + 4 training · image not yet built |
| **Evaluations** | issue |
| **Recorded runs** | 1 defective + 1 control; 200 steps each; training progress timers 33:43:44 / 25:51:05; lossless gzip logs restored before use |

## Difficulty

In `verl_tool/agent_loop/verltool_agent_loop.py`, `VerlToolAgentLoop.run()` reconstructs the first `<search>...</search>` action without `<|im_end|>`. The subsequent `new_gen_tokens[:-1] + gen_ids[-1:]` assumes the reconstructed final token is an EOS; instead it removes the `>` ending `</search>`. With the captured tokenizer this becomes `</search<|im_end|>`. The tensors remain valid, tools may still return observations, and training reward initially improves. In the original full batches, step 1 has 124/128 defective rollout texts with unbalanced search tags, versus 7/128 control texts; at step 200 the counts are 123/128 and 0/128. These full-batch counts are recorded in the hidden oracle's `rollout_full_batch_summary.json`. The shipped JSONL files now retain only three rows per step (first, seeded random interior, last), at steps 1/40/80/120/160/200: 18 rows per arm.

| Step-200 measurement | Defective | Control |
| --- | ---: | ---: |
| `critic/score/mean` | -1.0 | 0.6796875 |
| `response_length/clip_ratio` | 0.5625 | 0.0 |
| `verl_tool/is_traj_finished/mean` | 0.0 | 1.0 |
| Length-exhausted / final-answer trajectories | 128 / 0 | 0 / 128 |

## Reference solution

- **issue**: identify the reconstruction line and its interaction with the subsequent last-token substitution. Link corrupted delimiters to malformed trajectories and eventual collapse. General claims about bad data, insufficient model size, broken infrastructure, or no learning are insufficient; the defective run reaches positive training reward before collapsing.

## Verification

- **issue**: the standard LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence. Passing requires all three: verdict exactly matching the oracle, root-cause score above 0.8, and evidence score above 0.8; a fully wrong verdict cannot pass on the other two alone.
