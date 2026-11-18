# GRPO actor and rollout probabilities diverge

Training DeepSeek-R1-Distill-Qwen-7B with GRPO on Skywork-OR1-RL-Data (7B difficulty 1–15, prompt <= 1024); actor and rollout log-probabilities diverge on the same generated tokens.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `probability-mismatch` `output-support` `policy-gradient` |
| **Framework** | veRL 0.9.1 (GRPO), vLLM 0.24.0, PyTorch 2.11.0 |
| **Run time** | ~7 minutes (step 1) and ~11 minutes (step 2) in shared-image qualification; 275 steps in historical continuation |
| **Resources** | 2 GPUs |

## Difficulty

The vLLM rollout masks token IDs outside the tokenizer vocabulary before sampling and computing log-probabilities, whereas the FSDP actor normalizes log-probabilities and entropy over the full LM head output dimension. Because the tokenizer covers 151,665 IDs while the LM head has 152,064 rows, the 399 unreachable padded rows can accumulate probability mass. That extra mass shifts the actor's log-probability for every valid sampled token and exposes unreachable actions to policy gradients.

The frozen historical continuation spans 275 recorded steps (726–1000), using 64 prompts with 8 responses each and four 128-response optimizer minibatches per step. On three selected 16,384-token responses, the mean absolute rollout-versus-actor log-probability differences are 4.873, 4.869, and 4.768. In a seven-position shared-image replay containing raw logits and both log-probabilities, the unreachable-token mass reconstructed from the logits reaches 0.9955–0.9974 at later positions. Separate backend numerical discrepancies and a conditional reward-worker recovery bug have distinct evidence documented in the oracle.

A separate two-step qualification run on the shared image used 8 prompts × 8 responses, one 64-response optimizer minibatch, a 16,384-token limit, and the native V1 synchronization path, starting a fresh optimizer from private `checkpoint-975` weights. Step 1 ran at zero warmup learning rate (leaving weights unchanged, with the next logged scheduler rate at 0.0000001) and saved a complete checkpoint (`CP1`). Resuming from `CP1`, step 2 ran at learning rate 0.0000001 across 64 responses (43 positive and 21 negative rewards, with 5 of 8 groups mixed), produced a gradient norm of 27.8426, recorded Adam step 2 on both ranks, changed all four checked parameter slices, and saved `CP2`. This short qualification is distinct from the historical continuation, whose starting `checkpoint-725` is not distributed. The issue workspace provides the historical logs, the seven-position replay, and the shared-image qualification scalar logs plus three responses per step under `experiment_logs/shared_image/`, and uses `scripts/train.sh --check` for setup verification.

## Reference solution

- **issue**: identify the mismatch between the rollout's tokenizer-sized support mask and the actor's full-vocabulary log-probability and entropy normalization, and explain its algebraic effect on valid-token log-probabilities and policy gradients. Describe a repair that aligns the active support while preserving padded checkpoint and optimizer tensor shapes. Credit independently supported backend numerics or reward-worker findings separately.

## Verification

- **issue**: the official LLM judge compares `agent_answer.md` against the hidden oracle across verdict, root cause, and evidence (weighted 0.2 / 0.6 / 0.2 for overall reporting). Passing requires a verdict score of 1.0 and both root-cause and evidence scores strictly above 0.8. This is an issue-only task without a calibrated solution accuracy threshold; the frozen logs and seven-position replay allow full verification without the private checkpoint.
