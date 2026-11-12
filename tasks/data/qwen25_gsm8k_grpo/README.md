# Qwen2.5 GSM8K GRPO run

Training Qwen2.5-1.5B-Instruct with GRPO on GSM8K completes 40 steps, but a preprocessing regression exposes the current problem's gold answer in about 20% of training prompts and reinforces dependence on that answer.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `input-contamination` `data-preprocessing` |
| **Framework** | verl 0.9.0 at 483b8a009ba3a97563edee3a19887e4862b8094a; GRPO |
| **Run time** | Historical training 45.54 minutes, excluding queue |
| **Resources** | Historical training 1 H800; issue evaluation 0 GPU |

## Difficulty

Nothing crashes in the recorded run. The native reward function scores correctly, and 160 observed AdamW updates change sampled parameters. The preprocessor appends the same row's answer to 1,494 of 7,473 training prompts; all labels, row order and 1,319 test rows remain unchanged. This is a controlled reproduction of a plausible data-pipeline regression, not a discovered upstream verl incident.

The affected subset's online accuracy rises from 9.5% at step 1 to 96.0% at step 40. A real rollout computes 55/6≈9.17, then explicitly adopts the reference answer 11. Clean test accuracy reaches 78.09%; neither heldout stagnation nor clear response-length collapse was observed. Independent author evaluation found wrong-hint following 74.02% after this run versus 16.60% after matched no-leak training. That comparison used one paired training seed; it is attribution evidence, not a solution threshold.

The agent receives only the defective side: complete scalar metrics, training dumps at 1/20/40, clean validation at 0/20/40, and the final clean test. Targeted author intervention panels, control-side source/results and root-cause reports are excluded. Original numeric metrics, prompts, completions and scores are preserved; private transport/path labels are normalized. No live agent-difficulty result is claimed.

## Reference solution

- **issue**: identify the label-to-prompt concatenation in `verl/examples/data_preprocess/gsm8k.py::process_fn`, establish that it is the current row's gold answer, and connect it to actual answer adoption in the frozen rollout. The native scorer is correct. KL, entropy, format learning alone, or claiming that training never improved is insufficient.

## Verification

- **issue**: unchanged family LLM judge and instruction. Reported weights are verdict/root cause/evidence = 0.2/0.6/0.2; passing requires verdict = 1 and both other dimensions > 0.8. No new training or GPU is used.

Assets are downloaded during prepare, not baked into Docker or shipped in GitHub/source archives. Public dataset: [LIF1014/qwen25-gsm8k-grpo](https://huggingface.co/datasets/LIF1014/qwen25-gsm8k-grpo), pinned at `1d60021828cd3d735056165ffcb587408caa16ac`. Prepare downloads it to `/workspace/data/snapshot`, materializes raw inputs at `/workspace/data/gsm8k`, and regenerates `/workspace/data/preprocessed/{train, test}.parquet`. Every runtime input is checked by SHA256. Model: official `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The environment references the existing digest-pinned shared base image and creates a persistent dependency overlay during prepare. The previous package's fresh 3-step GPU smoke passed 12 parameter-changing updates with finite gradients and zero reward mismatches; that is historical runtime qualification. The remote pool has no Docker daemon and restricts registry prefixes, so no direct remote Docker build or Docker Hub pull is claimed. The HF data update requires no new GPU training.

Updated HF prepare qualification: a fresh remote prepare anonymously downloaded the complete public dataset snapshot at the pinned commit,verified every asset SHA256,and regenerated rows identical to the recorded input snapshot. This verifies asset preparation without claiming a remote Docker build.
