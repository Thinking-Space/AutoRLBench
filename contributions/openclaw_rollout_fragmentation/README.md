# OpenClaw rollout fragmentation: one terminal rollout, 31 training fragments

Training Qwen3.5-4B with Slime GRPO on OpenClaw/ClawGym tasks produced a recorded terminal rollout with 31 training fragments, 3,043,397 sequence tokens and 79,731 effective mask tokens.

**Draft raw-material contribution, not a validated AutoRLBench task.** The contributor and maintainer discussed accepting trajectory material for this efficiency case. This directory is deliberately outside `tasks/`: no shared-image reproduction, exact historical working-tree identity, repaired control or evaluation threshold is asserted.

| | |
| --- | --- |
| **Author** | Pengcheng Huang |
| **Affiliation** | Northeastern University |
| **Type** | Infrastructure / efficiency investigation |
| **Domain** | Tool-use RL |
| **Tags** | `trajectory-fragmentation` `token-replay` `long-context` `agent-rl` |
| **Framework** | Slime (GRPO), OpenClaw, historical Qwen3.5-4B recipe |
| **Run time** | Selected episode: 1528.2952196598053 seconds |
| **Resources** | Candidate recipe: 32 GPUs (8 actor + 24 rollout); original device type not established by this contribution |

## Recorded evidence

The selected batch is rollout batch 187. The full private source dump was read to compute these batch statistics; only the selected episode is shipped:

| Scope | Terminal rollouts | Fragments | Sum of sequence lengths | Effective mask tokens |
| --- | ---: | ---: | ---: | ---: |
| Whole batch | 128 | 480 | 30,529,345 | 3,879,465 |
| Selected episode | 1 | 31 | 3,043,397 | 79,731 |

Batch mean fragmentation is 3.75; the maximum is 31. The selected episode has a maximum individual sequence length of 115,781 tokens, 63 agent turns, 83 tool calls and 64 model requests. All 31 fragments are marked completed and have `remove_sample=false`. The agent exited 0, but its verifier failed and the recorded reward is 0. This is efficiency evidence, not a successful-answer example.

These are sequence-length sums, not a measured count of physical GPU forward passes, FLOPs or a demonstrated speedup. The 31 fragments are training segments, not 31 spawned subagents. A long prefix can be necessary context; these counters alone do not prove that every split was avoidable.

`evidence/episode.json` preserves all 31 fragment masks, logprobs, rewards, response lengths and sanitized response text. Original prompt text and token-ID arrays were deliberately omitted to avoid publishing private paths encoded in tokens. `evidence/fragment-lengths.csv` gives a compact per-fragment view. The batch statistics come from the complete source dump; no claim is made that the full batch is included here.

## Source and diagnosis boundaries

`code/` is a minimized, sanitized historical candidate from commit `0925c7980f2906d97814757ff8d573ec0c02f0b8`, immediately preceding the recorded launch. The original runtime did not record its Git SHA or uncommitted edits. The actual runtime trace and original job entry identify the repository/launcher family, but this snapshot is not authenticated as the exact running tree. The relevant trajectory/OpenAI adapter files have no committed changes in the inspected start-to-dump interval; transient uncommitted changes remain possible.

In the candidate, `slime/agent/trajectory.py` distinguishes `CLEAN`, `REALIGN` and `FORK`; prefix drift can close one token builder and open another. Splitting protects token/logprob alignment. The `OpenClawAdapter` explicitly preserves reasoning, parallel tool calls and mixed content, so **the generic OpenAI adapter defaults must not be misreported as the active OpenClaw policy**. This evidence does not establish a unique cause for all 31 boundaries; candidate explanations include token replay/serialization drift, message-tree divergence and genuine client branching. Full original context identifiers are not available in the public material to replay those decisions conclusively.

The intended follow-up is to instrument and reproduce the fragmentation on a public model/shared environment and separate necessary splits from avoidable replay drift. Simply disabling splitting, inflating the threshold or retokenizing outputs without preserving logprob alignment is not a demonstrated repair.

## Packaging status and privacy

Issue-only material is proposed. There is no fixed workspace, repaired control or calibrated solution threshold. The historical run used a prior runtime whose membership in the benchmark's shared-image set has not been established. No new GPU rerun or paid judge evaluation was performed. For that reason this draft does not invent `task.toml`, a digest pin or a passing harness-validation result.

Unrelated experiments, internal model recipes, private plugins, platform submission and credential-transfer helpers are omitted. Hardcoded credentials, internal endpoints, personal paths, runtime identifiers and absolute wall-clock timestamps were removed/anonymized. Original numeric measurements, relative elapsed durations, masks and logprobs were retained. Source changes are recorded by before/after hashes in `sanitization-manifest.json`; original private material is not embedded in this PR.

The historical launch script is reference material, not a ready-to-run deployment template. Private defaults are placeholders, only the observed static reward path is retained, and secret variables are excluded from Ray JSON forwarding. A real replay needs explicit model/data/runtime configuration, a memory-only credential-delivery integration and a fresh environment check. `SHA256SUMS` identifies the shipped files.
