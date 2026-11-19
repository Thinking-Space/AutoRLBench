# Claude Code native WebSearch on a self-hosted RL backend

Training Qwen3.5-9B with GRPO on Search-R1 NQ/HotpotQA questions inside Claude Code produces normal actor generations and optimizer steps, but every recorded native WebSearch HTTP request fails against the self-hosted inference backend.

| | |
| --- | --- |
| **Author** | Pengcheng Huang |
| **Affiliation** | Northeastern University |
| **Type** | Infrastructure |
| **Domain** | Search RL / Tool-use RL |
| **Tags** | `native-websearch` `server-tool-compatibility` `self-hosted-inference` `agent-rl` |
| **Framework** | Slime (GRPO), SGLang 0.5.15.post1, Torch 2.11.0+cu129, Claude Code 2.1.300 |
| **Run time** | ~11 minutes including startup; ~4.5 minutes of CC sampling and training |
| **Resources** | 8 × H100 (80 GB), recorded on two 4-GPU nodes |

## Difficulty

A server that can generate the model's next tokens does not necessarily implement the server-side tools expected by the client. Ordinary CC Messages calls are translated by the token-capturing adapter into real SGLang `/generate` requests. Native search instead carries an Anthropic `web_search_20250305` declaration and forced tool choice to SGLang's Messages converter, which skips the built-in and returns HTTP 400. In this recorded CC version the original request path is `/v1/messages?beta=true`, **not `/search`**. Forwarding rewrites the model alias and does not preserve every query/header; the original and forwarded bodies are both recorded.

All **891** recorded native search HTTP responses are 400. There are **895** outer CC WebSearch calls and **300** background outcomes, **299** with trainable tokens; these are different counters. Only **96** samples were selected for training, with **25,190** effective actor tokens. All selected token IDs and logprobs match the original SGLang outputs. The three training batches contain 32 samples each:

| Zero-based step | Mean raw reward | Gradient norm | Prompt groups with reward variation |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 0.025 | 2.405400311945096 | 3 |
| 2 | 0.06875 | 1.7096229973075858 | 2 |

The final checkpoint-save log reports success at zero-based iteration **2**. Ray returned success, but the platform task ended **Failed** because the summary script indexed `trainable_tokens` on one shutdown outcome without that field. NCCL connection-reset messages also appear after the successful save, during final synchronization/disposal. The three train debug dumps were corrupt and are omitted; the three readable rollout dumps, source, complete run log and original token records support the diagnosis. There is no held-out evaluation or convergence claim. W&B was disabled.

The public image is the shared Slime environment used for the recorded run through its platform mirror. The task supplies the frozen Slime source and a pure-Python mbridge overlay. The recorded topology was four actor GPUs (TP2/DP2) plus four rollout GPUs (two TP2 engines); `run.sh` stages the same 4+4 layout on a single eight-GPU host. That relocated single-host packaging entry has not been GPU-rerun. Model arguments are the unchanged Qwen3.5-9B reference script; LR 1e-6, GBS32, BF16, three batches of eight prompts with four samples each, response cap 2048 and context 8192. QA reward is 1 for normalized exact match, 0.2 for a nonempty tagged wrong answer, and 0 otherwise; it does not grant a separate bonus for failed search and is not the full original XML-format reward.

## Reference solution

- **issue**: identify the boundary between client-native Anthropic search and a model-only serving backend: CC sends the native server-tool request through its configured base URL, while SGLang cannot execute that built-in and rejects forced selection after filtering the tools. Trace `code/cc_search_rollout.py`, the recorded native request/response bodies, and the installed SGLang Anthropic converter. Distinguish the primary incompatibility from postprocessing/shutdown problems. A missing `/search` route, broken weight loading, zero gradients in every step, or failure to start RL are incorrect diagnoses.

Issue-only: no repaired control run or outcome threshold was calibrated, and several provider/routing repairs are possible. One run was recorded; there is no estimate of cross-run variance.

## Verification

- **issue**: the copied LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted **0.2 / 0.6 / 0.2**, passing at **0.8**. Source attribution, all 891 HTTP records and the 96-sample token audit are preserved. The packaging validator and scaffold byte comparisons are run locally; no fresh harness GPU rerun or paid judge evaluation is claimed.
