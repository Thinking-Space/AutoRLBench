# GRPO credit assignment drifts during multi-turn tool rollouts

A complete Slime GRPO run shows how tool timeouts can change the sign of the learning signal for completed trajectories.

| | |
| --- | --- |
| **Author** | Pengcheng Huang |
| **Affiliation** | Northeastern University |
| **Type** | Algorithm |
| **Domain** | Agentic tool-use RL |
| **Tags** | `grpo` `reward-normalization` `tool-use` `aborted-rollouts` |
| **Framework** | Slime v0.3.2, SGLang 0.5.15.post1, Megatron Core 0.16.0rc0 |
| **Run time** | About 11 min 23 s from the first timestamped checkpoint conversion output to Ray job completion |
| **Resources** | 1 × NVIDIA H800 (80 GB) |

## Difficulty

`workspace/run.sh` uses the bundled Slime implementation to convert the pinned Qwen3.5-0.8B checkpoint and launch `workspace/slime/train.py` through Ray. SGLang samples multi-token assistant replies. The rollout feeds ticket catalog responses back into the policy context as masked observations. Megatron runs clipped GRPO and saves distributed checkpoints. The experiment runs 16 rollouts with four prompts and four samples per prompt, producing 32 optimizer updates. Slime evaluates before training and after every four rollouts. The offline generator creates 128 training tickets and 16 held-out tickets with disjoint IDs. The Slime snapshot retains its Apache-2.0 license.

This reproduces the task-reward portion of the historical OpenClaw defect in a local tool environment. The adapter introduces a repeatable timeout schedule on selected training tool requests. Held-out evaluation uses the catalog without this schedule. The pinned public framework supplies sampling, rollout batching, optimization and checkpointing.

| Held-out evaluation | Historical callback | Eligibility control |
| --- | ---: | ---: |
| Before training | 12/16 | 12/16 |
| After 4 rollouts | 15/16 | 16/16 |
| After 8 rollouts | 16/16 | 15/16 |
| After 12 rollouts | 0/16 | 2/16 |
| After 16 rollouts | 0/16 | 0/16 |

The two runs share the checkpoint, prompts, seed, budget and timeout schedule. The control changes the callback's normalization population. Sampled replies diverge between runs. The final scores show that this pair cannot establish whether the normalization bug caused the held-out decline.

The recorded run reached all 32 optimizer updates. Its 256 training trajectories included 27 masked tool timeouts. Recomputing GRPO on the same batches finds 81 eligible trajectories whose advantages change when timeout records are excluded, including four sign flips. In rollout 0, group 1 has rewards `[0.0, 1.0, 0.6, 0.6]`. The leading zero is a masked timeout. Each 0.6-reward completion has advantage **+0.1213** in the recorded calculation and **−0.5773** when the same batch is normalized over eligible completions.

## Reference solution

- **issue** — Identify how `routing.reward.post_process_rewards` includes interrupted records in group normalization before Slime applies the training loss masks. Normalize eligible terminal trajectories before broadcasting their advantages to samples. Keep legitimately scored zero rewards in the population. A late mask leaves the peer advantages computed from the wider population.

This contribution is issue-only because the paired runs both finish at 0/16, while the recorded groups directly expose the incorrect learning signal.

## Verification

- **issue** — The defect was exercised in the declared shared Slime image. `workspace/logs/training.log` retains the full training command's combined stdout/stderr, including checkpoint conversion, Ray and SGLang logs, per-step optimizer metrics, evaluation and checkpoint saves. W&B was disabled. The rollout snapshots from rounds 0, 7 and 15 record actual generated text and tool outcomes. Compare their terminal reward populations with the callback that prepares rewards for the actor.

The prepare stage materializes the pinned model into `assets/model`. From the workspace, `./run.sh assets/model outputs` reproduces the recipe on one compatible GPU. The launcher builds the derived Megatron checkpoint through the GPU converter before starting Ray. The recorded platform run used stochastic SGLang sampling, so scores may vary across launches.
