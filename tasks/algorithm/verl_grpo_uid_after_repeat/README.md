# GRPO uid assigned after rollout repeat

Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The uid assignment
that groups rollout samples by prompt was moved to after the repeat step, making
every GRPO group a singleton and silently degrading training to baseline-free REINFORCE.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | algorithm |
| **Tags** | `verl` `grpo` `advantage` `uid` |
| **Model** | Llama-3.2-1B-Instruct |
| **Dataset** | DAPO-Math-17k |
| **Framework** | verl 0.9.0.dev @ `6f2aa24abfc01cc72811b376aeb3936503885096`, GRPO |
| **Environment** | `verl` (verlai/verl:vllm017.latest) · 2 × H800 for training |
| **Evaluations** | issue |
| **Recorded runs** | 1 defect run (60 steps, 2026-10-16) |

## Difficulty

The `uid` field is GRPO's grouping key: all responses sampled from one prompt
must share a uid so `compute_grpo_outcome_advantage` can compute the group mean
and standard deviation. Moving the assignment below `batch.repeat()` gives every
one of the `train_batch_size × rollout_n` rows its own fresh UUID. The
`len(group) == 1` branch of `compute_grpo_outcome_advantage` silently
substitutes `id2mean = 0.0` and `id2std = 1.0` without raising, so the group
baseline is never subtracted and GRPO becomes REINFORCE with no baseline and no
variance normalisation.

The held-out curve does not reveal the defect at a 60-step budget (0.0454 →
0.0570 vs. control 0.0592). The diagnostic signal is structural, not metric:

| metric | defect run |
| --- | --- |
| `critic/advantages/min` < 0 steps | 0 / 60 |
| `critic/advantages/max` on rewarded steps | always `0.9999990463` (fp32 `1/(1+1e-6)`) |
| `actor/pg_loss` range | [−0.067, 0] — never positive |
| `actor/entropy` first → last | 4.81 → 1.31 (vs. 4.42 → 2.99 in control) |

## Reference solution

- **issue**: identify that `batch.non_tensor_batch["uid"] = ...` was moved below
  `batch.repeat(repeat_times=rollout.n, interleave=True)`. The diagnosis must
  name the uid assignment's position relative to `batch.repeat`, not just report
  that group size is 1 or that advantages are not normalised.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires ≥ 0.8. If no judge model is configured,
  the verifier writes `status=unavailable` instead of a pass.

The default prepare step fetches Llama-3.2-1B-Instruct weights (~2.4 GB) and
the DAPO-Math-17k parquet files. The `verl` image includes vLLM 0.17, torch
2.10 / CUDA 12.9 / Ubuntu 22.04. GPU training requires 2 × H800.
