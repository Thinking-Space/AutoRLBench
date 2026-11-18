# Excluded OpenClaw rollouts contaminate GRPO advantages

A synthetic CPU replay of an OpenClaw Slime GRPO reward callback shows an excluded zero-reward rollout flipping a valid rollout's advantage from negative to positive.

| | |
| --- | --- |
| **Author** | Pengcheng Huang |
| **Affiliation** | Northeastern University |
| **Type** | Algorithm |
| **Domain** | Tool-use RL |
| **Tags** | `grpo` `reward-normalization` `aborted-rollouts` `agent-rl` |
| **Framework** | Slime OpenClaw reward grouping and Torch GRPO normalization |
| **Run time** | Seconds for six synthetic batches |
| **Resources** | CPU only, 0 GPUs |

## Difficulty

A rollout can already have a zero loss mask and still influence other rollouts. The historical cost-budget callback grouped terminal rollouts by prompt and passed the entire prompt group to `task_advantages`. Its abort path emitted a zero reward and zero loss mask, along with the exclusion flags. It received failure records marked `remove_sample` or `ABORTED` alongside scored task completions. By the time the interrupted rollout was excluded from training, its zero reward had already contributed to the group mean and sample standard deviation. One terminal rollout may yield several fragments, so eligibility and normalization apply to terminal rollouts before the resulting values are expanded back to fragments.

| Synthetic batch | Advantage of the valid 0.6-reward rollout, observed | Filtered normalization control |
| --- | ---: | ---: |
| Valid rewards `[1.0, 0.6]` | −0.707104 | −0.707104 |
| Add a `remove_sample` zero | +0.132453 | −0.707104 |
| Add an `ABORTED` zero | +0.132453 | −0.707104 |
| Add a legitimate completed zero | +0.132453 | +0.132453 |

The case uses the original float32 Torch normalization and terminal-rollout grouping helpers with a minimal sample interface. The replay reduces the historical callback to the task-advantage path. It runs on CPU from the supplied fixtures. The effective advantage field shows the consequence of masking after normalization. It measures the training signal entering optimization. The recorded evidence comes from a controlled re-execution of the historical task-advantage path on synthetic inputs.

## Reference solution

- **issue**: identify that excluded records enter `task_advantages` through the unfiltered prompt groups in `workspace/replay.py`. Explain how the premature zero shifts the mean from about 0.8 to 0.5333, changing the sign for the valid 0.6-reward rollout. Preserve legitimate scored zeros and one normalization contribution per terminal rollout. A late mask suppresses training on the interrupted fragments but cannot restore the valid peers' advantages.

Issue-only because the paired replay isolates the task-advantage math. It provides no repaired end-to-end training or calibrated training threshold.

## Verification

- **issue**: use the official judge scaffold with verdict / root cause / evidence weights of 0.2 / 0.6 / 0.2 and a pass mark of 0.8. The snapshot includes six batches and their observed CPU results. The paired filtering check also verifies a singleton valid rollout, a fully excluded group, and fragment expansion. Run `python3 workspace/replay.py` in the declared Slime image to reproduce the observed arm.
