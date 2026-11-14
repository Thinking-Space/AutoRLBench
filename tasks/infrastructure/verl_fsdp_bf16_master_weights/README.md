# bf16 master weights round away AdamW updates

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, the launcher sets the FSDP actor's `model_dtype` to bf16, so parameters and AdamW state are bf16 and lr-sized updates round away for most weights. The KL to the reference grows 5-13× more slowly and held-out accuracy ends at 0.4655 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | Meta |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `mixed-precision` `bf16-master-weights` `slow-learning` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~34 minutes |
| **Resources** | 1 × A100 (40 GB) |

## Difficulty

```bash
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
```

verl passes `model_dtype` to `from_pretrained`
(`verl/workers/engine/fsdp/transformer_impl.py::_build_module`). The default `fp32` keeps fp32 master
weights; compute is bf16 anyway through FSDP mixed precision. With bf16 a weight `w` moves only in
steps of about |w|/256. AdamW's step (≈ lr = 2e-6) is below half that spacing for any weight above
about 1e-3, so it rounds back.

Nothing reports it: gradient norm (0.60-0.70), loss, learning rate and throughput match the control,
no NaN. Only the drift is slow: entropy 0.48 → 0.24 (control → 0.08).

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out | 0.4625 | 0.4602 | 0.4640 | 0.4738 | 0.4761 | 0.4685 | 0.4655 |
| control held-out, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control held-out, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |
| defective `actor/kl_loss` | 0.0006 (step 1) | 0.0016 | 0.0041 | 0.0072 | 0.0114 | 0.0131 | 0.0140 |
| control `actor/kl_loss` | 0.0003 (step 1) | 0.0210 | 0.0380 | 0.0500 | 0.0579 | 0.0596 | 0.0697 |

Wrong diagnoses: learning rate or KL coefficient (same as the control), reward or data, capacity,
AdamW epsilon, bf16 compute itself (the control also computes in bf16).

## Reference solution

- **issue**: `model_dtype=bf16` → bf16 master weights and optimizer state → AdamW steps rounded away
  → slow drift, flat accuracy. "Learning is slow" or "lr too low" without the dtype earns partial
  credit.
- **solution**: drop the override (`fix.diff`); verl's default is fp32.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. The runner is not byte-compared (the fix is in it). Defective step 60: 0.4655 (best 0.4761); control 0.5580 / 0.5406 (seeds 42 / 43);
  margins 0.034 / 0.031. The control's step 0 is 0.4503, so a run that does not learn fails. One
  defective run and two control runs: margins, not a variance estimate. The regenerated training parquet
  is checked against the pinned raw split and `metrics.jsonl` must cover steps 0-60. Grader replay on the
  recorded evidence: defective exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest).
