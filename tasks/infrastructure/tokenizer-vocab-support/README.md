# GRPO actor and rollout probabilities diverge

Training DeepSeek-R1-Distill-Qwen-7B with GRPO on Skywork-OR1-RL-Data (7B difficulty 1-15, prompt limit 1024) exposes a systematic actor-versus-rollout log-probability shift on the same generated tokens.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `probability-mismatch` `output-support` `policy-gradient` |
| **Framework** | veRL 0.9.1 (GRPO), vLLM 0.24.0, PyTorch 2.11.0 |
| **Run time** | 30 h 11 min of fault allocation time across the accepted CP725-to-CP1000 segments |
| **Resources** | 2 GPUs |

## Difficulty

The rollout masks IDs outside the 151,665-token tokenizer support. The FSDP actor normalizes log-probability and entropy over all 152,064 output rows. The 399 padded rows can absorb probability mass and shift the likelihood of valid sampled tokens. The seven-position shared-image replay in `workspace/experiment_logs/` reconstructs padded mass of 0.9955-0.9974 at later positions. The predicted shift matches the measured shift within 0.0101 nat. Sparse same-token errors remain after vocabulary alignment and are covered separately in the issue oracle.

The original 275-update historical trace supplies diagnosis evidence. The graded continuation targets the original 1,000-step schedule and restores the full native CP725, including the optimizer and data cursor. It uses 64 prompts with eight responses per prompt, a 16,384-token response budget, and four optimizer updates per training step. The harness retains CP990 and CP1000. The original trainer saves, evaluates and returns at the final step.

## Reference solution

- **issue** Diagnose the inconsistency between the rollout support mask and the actor's likelihood and entropy normalization. Explain its effect on the normalization constant and policy gradients, and distinguish the systematic shift from residual numerical discrepancies.
- **solution** Restrict actor logit normalization and related entropy calculations to the tokenizer-valid support on the padded and remove-padding paths. Keep checkpoint and optimizer tensor shapes and the configured training workload. The reference patch guards the inactive fused path explicitly.

## Verification

- **issue** The LLM judge compares `agent_answer.md` with the hidden oracle across verdict, root cause, and evidence. It reports a weighted score of 0.2/0.6/0.2. Passing requires verdict 1.0 and both root-cause and evidence scores strictly above 0.8.
- **solution** The agent submits `report.md`. The harness restores CP725 and runs updates 726 through 1000. After the native checkpoints pass validation, the scorer recomputes answer-level accuracy from AIME24, AIME25, and AMC23, with 32 draws per question and 3,200 responses in total. The final metric is `val-core/overall/acc/mean@32`. The cutoff is 2,162 correct (67.5625%). The common start answered 2,137 correctly, the fault run ended at 2,012, and the repair ended at 2,173. The repair cleared the cutoff by 11 answers and the fault fell short by 150. The repair gained 5.03 percentage points over the fault and 1.13 over the start, with question-paired 95% intervals [2.94, 7.28] and [0.03, 2.25]. The cutoff uses one segmented native training run per arm. The harness allows up to 35 hours for a fresh rerun from CP725.
