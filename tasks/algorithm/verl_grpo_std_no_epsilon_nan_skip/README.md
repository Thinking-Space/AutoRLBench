# GRPO std without epsilon: every step NaN and skipped

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, the advantage is divided by the group std without epsilon. Groups with identical rewards give NaN advantages, every optimizer step is skipped, and held-out accuracy stays at 0.4564 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | NTU |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `nan-advantage` `skipped-update` `silent-failure` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~36 minutes |
| **Resources** | 1 × A100 (80 GB) |

## Difficulty

`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`:

```python
scores[i] = (scores[i] - id2mean[index[i]]) / id2std[index[i]]   # upstream: / (id2std[index[i]] + epsilon)
```

`epsilon` is still a parameter, unused. 18-21 of 64 groups per step have eight equal rewards → 0/0.
The token-mean loss and the gradient norm are NaN, and verl's FSDP engine zeroes the gradient and
skips the step (`WARN: grad_norm is not finite: nan`, 120 times = 2 mini-batches × 60 steps).

Metrics that do not touch the advantage look normal: training score 0.30-0.39, entropy 0.44-0.54,
length 294-327 tokens. The giveaways: `actor/pg_loss`, `actor/grad_norm` and `critic/advantages/*`
are NaN at every step (`null` in `metrics.jsonl`), and `actor/kl_loss` is exactly 0.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out | 0.4617 | 0.4723 | 0.4678 | 0.4587 | 0.4526 | 0.4602 | 0.4564 |
| control held-out, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control held-out, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |
| defective training score | n/a | 0.359 | 0.320 | 0.311 | 0.328 | 0.389 | 0.365 |
| control training score | n/a | 0.541 | 0.559 | 0.647 | 0.580 | 0.658 | 0.693 |

Wrong diagnoses: bf16/FSDP numerics or the learning rate (the NaN is made before the model; the
control is stable), exploding gradients (NaN from step 1, not large), reward or data, "zero-variance
groups are normal" (they are; upstream gives them a zero advantage).

## Reference solution

- **issue**: the missing epsilon → NaN advantages → NaN loss and gradient → every step skipped → the
  policy never changes. "NaN loss" without the division earns partial credit.
- **solution**: restore `+ epsilon` (`fix.diff`); turning std normalization off is equivalent.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. Defective step 60: 0.4564 (best 0.4723); control 0.5580 / 0.5406 (seeds 42 / 43);
  margins 0.038 / 0.031. The control's step 0 is 0.4503, so a run that does not learn fails. One
  defective run and two control runs: margins, not a variance estimate. The regenerated training parquet
  is checked against the pinned raw split and `metrics.jsonl` must cover steps 0-60. Grader replay on the
  recorded evidence: defective exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest). The defective run used an
A100 80 GB; control seed 42 an A100 40 GB, seed 43 an A100 80 GB.
