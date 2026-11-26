# GRPO rollout and training length disagreement

During GRPO training on AIME 2024, the trainer counted every answer from Nemotron 4B as exactly 16,384 tokens long, even when the model had already finished answering.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `rollout-accounting` `token-credit` |
| **Framework** | veRL, vLLM 0.24.0 |
| **Run time** | 13 min 17 s in the harness training phase |
| **Resources** | 1 GPU |

## Difficulty

The model sampled four answers per problem. Each batch contained eight problems, and the scorer found both correct and incorrect answers. Yet the training log reported the maximum response length for every answer in both batches.

| Training step | Correct answers | Mean tokens in exported text | Answers with text below the cap | Mean tokens counted by trainer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14/32 | 7,368.53 | 25/32 | 16,384 |
| 2 | 12/32 | 9,764.97 | 20/32 | 16,384 |

The text lengths come from re-encoding the exported answers, which omit special tokens. A separate archived generation preserves the exact token IDs: the engine returned 2,911 tokens for a correct answer, while the trainer counted 16,384. Understanding how the score could still be correct requires following that answer through both grading and the policy update.

## Reference solution

- **issue**: The trainer used the checkpoint's end-of-sequence ID (2) to find the end of each padded response. The chat tokenizer also allowed a response to end with ID 11, which the rollout code used for padding. When an answer ended with 11, the trainer could mark the entire padded row as valid. The scorer skipped special tokens and read the correct answer, but placed its reward at the last position marked valid. GRPO then applied the answer's advantage to the padding as well. Building the mask from the length returned by the engine preserves the actual response boundary.

## Provenance

The vendored rollout code is the recorded behavior of the pinned OPQAD veRL fork at the commit listed in `environment/source_manifest.json`; the checkpoint-versus-chat-tokenizer EOS mismatch it exhibits is not claimed to be a bug in upstream huggingface/verl. The controlled replay figures below are author-side measurements; only the shipped logs, rollouts, and engine records are agent-verifiable evidence.

## Verification

The official harness completed both GRPO updates. In a controlled replay of the 2,911-token answer, the score stayed at 1 under both masks. Using the returned length moved the reward from position 16,384 to position 2,911. For two answers to the same problem, one correct and one incorrect, the original mask gave 6,521 padding positions a nonzero gradient with respect to token log probability. The length-based mask reduced that count to zero.

- **issue**: The judge scores `agent_answer.md` against the hidden diagnosis. Verdict, root cause, and evidence carry weights of 0.2, 0.6, and 0.2. The passing score is 0.8. This issue evaluates diagnosis of the misplaced training signal. Held-out accuracy stayed at 4/16 before training and after each update.
