# Integer ground truth never matches the string scorer: zero reward

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, data preparation stores each ground truth as an int while verl's GSM8K scorer compares it with the extracted answer string. Every reward is 0, no gradient is produced, and held-out accuracy stays at 0.4602 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | Meta |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `type-mismatch` `zero-reward` `silent-failure` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~34 minutes |
| **Resources** | 1 × A100 (40 GB) |

## Difficulty

`data_prep/gsm8k.py`:

```python
"reward_model": {"style": "rule", "ground_truth": int(float(extract_solution(answer)))},
```

`gsm8k.compute_score(..., method="flexible")` returns 1.0 only if `answer == ground_truth`, and
`"21" == 21` is `False`. Every reward, advantage, `actor/pg_loss`, `actor/grad_norm` and
`actor/kl_loss` is exactly 0 for 60 steps.

The logged `val-core/openai/gsm8k/acc/mean@1` is 0.0 at every evaluation, which reads like a model
that cannot do GSM8K. It can: 648 of 1,536 sampled responses end with the right number (e.g.
"... = **21 gifts.**" with `gts` 21, scored 0.0). The only visible difference in the dumps is
`"gts": 21` where the control has `"gts": "21"`.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective logged val acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| defective held-out | 0.4602 | 0.4632 | 0.4549 | 0.4632 | 0.4564 | 0.4701 | 0.4602 |
| control held-out, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control held-out, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |

Wrong diagnoses: capacity, the `####` format (no format is required), the extraction regex (it
extracts the right number; the comparison fails), lr/KL/GRPO settings, vLLM.

## Reference solution

- **issue**: the int cast vs the scorer's string equality → every reward 0 → no gradient; the logged
  0.0 accuracy is the same mismatch. "Reward is always zero" without the type earns partial credit.
- **solution**: store the string label (`fix.diff`); a numeric comparison in the reward is equivalent.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. Defective step 60: 0.4602 (best 0.4701); control 0.5580 / 0.5406 (seeds 42 / 43);
  margins 0.040 / 0.031. The control's step 0 is 0.4503, so a run that does not learn fails. One
  defective run and two control runs: margins, not a variance estimate. The regenerated training parquet
  is checked against the pinned raw split and `metrics.jsonl` must cover steps 0-60. Grader replay on the
  recorded evidence: defective exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest).
