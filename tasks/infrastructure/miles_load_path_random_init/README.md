# Wrong MEGATRON_MODEL_PATH causes Megatron to silently train from random weights

Training Qwen3.5-9B with arena-rubric GRPO using miles. `MEGATRON_MODEL_PATH` points to
the `release/` subdirectory rather than the parent directory containing
`latest_checkpointed_iteration.txt`. Megatron silently skips checkpoint loading, logs
one line saying it will start from random, and continues training with random weights.

| | |
| --- | --- |
| **Author** | xuxiaoang |
| **Affiliation** | Modelbest |
| **Type** | Infrastructure |
| **Domain** | Arena-Rubric RL |
| **Tags** | `checkpoint` `random-init` `silent-failure` `load-path` `megatron` |
| **Framework** | miles (GRPO), Megatron train + SGLang rollout + Ray |
| **Run time** | Full training completes; defect spans all training steps |
| **Resources** | 8 GPU |

This document is for maintainers and reviewers.
Evidence is reconstructed from session documentation.

## Difficulty

The defect is a one-word path difference: `MEGATRON_MODEL_PATH` is set to
`/path/to/Qwen3.5-9B_torch_dist/release` (a subdirectory), when it should be
`/path/to/Qwen3.5-9B_torch_dist` (the parent directory containing
`latest_checkpointed_iteration.txt`).

Megatron's `checkpointing.py` checks for `latest_checkpointed_iteration.txt` in
the given path. When this file is absent, it logs exactly one line:

```
will not load any checkpoints and will start from random
```

Then it continues. The job runs to completion with no error. Because SGLang
synchronizes weights from the Megatron model at every rollout, all rollout
responses are from a randomly-initialized model. The reward curve stays near zero,
but this looks like "the model hasn't converged yet" in the first few rollouts.

The only ways to catch this are:
1. `grep "start from random"` in the training log
2. The `--load` path does not contain `latest_checkpointed_iteration.txt`
3. Per-step reward distribution is inconsistent with a pretrained model (essentially 0 from step 0)

A correct launch adds a fail-fast check: `if [ ! -f "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt" ]; then exit 1; fi`

## Reference solution

- **issue**: identify that `MEGATRON_MODEL_PATH` points to `release/` (a subdirectory)
  instead of the parent directory, that Megatron therefore cannot find
  `latest_checkpointed_iteration.txt`, and connect this to the "will not load any
  checkpoints" log line and the near-zero reward throughout training. Reject "the model
  did not converge" and "the rubric judge failed" as root causes.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  `verdict_score == 1.0`, `root_cause_score > 0.8`, and `evidence_score > 0.8`.

issue-only: no post-fix run recorded; fix is a path correction with no post-fix
training log to grade.
