# Teacher log-probs shifted by one token collapse on-policy distillation

Distilling Qwen2.5-1.5B-Instruct into Qwen2.5-0.5B-Instruct on GSM8K with verl's on-policy distillation, the teacher's log-probs are shifted one token against the student's. The per-token advantage becomes noise, entropy climbs to uniform over the vocabulary, and held-out accuracy falls from 0.4663 to 0.0 by step 20; the aligned run ends at 0.4981.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | NTU |
| **Type** | Algorithm |
| **Domain** | Math RL (on-policy distillation) |
| **Tags** | `on-policy-distillation` `off-by-one` `policy-collapse` |
| **Framework** | verl 0.8.0 on-policy distillation (k1 log-ratio as the advantage), FSDP + vLLM, `run_opd_gsm8k.sh` |
| **Run time** | ~29 minutes |
| **Resources** | 2 × A100 (40 GB): student and teacher one GPU each |

## Difficulty

The student trains on its own rollouts. A vLLM teacher scores each rollout as a prompt
(`prompt_logprobs`), and `log p_student(y_t) - log p_teacher(y_t)` is the negative advantage of a
clipped policy-gradient update. The defect is in verl,
`verl/workers/rollout/vllm_rollout/utils.py::extract_prompt_logprobs`:

```python
    # NOTE: logprob of first prompt token is None; keep a zero placeholder for it so that
    # entry i lines up with token i of the sequence.
    prompt_logprobs_ls = [[0.0] * max(num_prompt_logprobs, 1)]
```

Upstream drops vLLM's leading `None` and pads a dummy entry at the end, so entry `i` holds the
log-prob of token `i+1`, the student's next-token layout. The workspace prepends a placeholder and
drops the end pad: same length, no shape error, but every teacher log-prob is one position off.

At step 1, before any update, `actor/distillation/abs_loss` is 0.72 (aligned run 0.25) and
`actor/grad_norm` is 22.8 (5.9). Entropy then goes 0.49 → 4.48 (step 5) → 9.36 (step 10) → 11.92
(step 60, uniform over 151,936 tokens), and responses fill the 512-token budget with text like
`oldem..,.,.,.,`.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out | 0.4663 | 0.0622 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| aligned held-out | 0.4572 | 0.4769 | 0.4678 | 0.4792 | 0.4875 | 0.4966 | 0.4981 |

Held-out = the last number of the greedy response, compared numerically with the gold answer. verl's
logged metric compares strings and misses answers such as "260." or "$26.00" (106 of 1,319 for the
aligned run at step 60).

Wrong diagnoses: learning rate or clipping, teacher quality, the k1 / policy-gradient formulation, a
template or tokenizer mismatch, truncation, the task reward. All are shared with the aligned run.

## Reference solution

- **issue**: the off-by-one in `extract_prompt_logprobs` → each log-ratio compares different tokens
  → the advantage is noise from step 1 → the student collapses. "Teacher and student are misaligned"
  without the extraction code earns partial credit.
- **solution**: not graded. Restore upstream's layout (the aligned run).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.

Issue tier only: the aligned run gains 4 points over its own step 0 (one seed), too little for a
repair threshold that a frozen student could not also pass. Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest) and the API-backed judge.
