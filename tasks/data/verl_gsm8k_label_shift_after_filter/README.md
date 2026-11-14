# One-sided question filter shifts 96% of GSM8K labels

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, data preparation drops long training questions from the question list but not from the answer list, so 96.5% of training rows are scored against another problem's answer. The training score stays near zero and held-out accuracy ends at 0.4443 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | Meta |
| **Type** | Data |
| **Domain** | Math RL |
| **Tags** | `label-misalignment` `data-preprocessing` `silent-failure` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~38 minutes |
| **Resources** | 1 × A100 (80 GB) |

## Difficulty

`data_prep/gsm8k.py`:

```python
questions = df["question"].tolist()
answers = df["answer"].tolist()
if split == "train":
    questions = [q for q in questions if len(q) <= MAX_QUESTION_CHARS]
rows = build_rows(questions, answers, split)   # zip(questions, answers)
```

`zip` truncates silently, so the row count (7,360) matches the control. From raw row 175 on, every
question carries a later problem's answer: 7,101 of 7,360 rows. The test split is not filtered and
stays correct.

The policy is fine; its reward is not. In the rollouts at steps 1, 30 and 60, 677 of 1,536 responses
answer their own question correctly, yet `critic/score/mean` averages 0.017 (0.000-0.063). Entropy
falls 0.51 → 0.23 on coincidental rewards while held-out accuracy does not move.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out | 0.4488 | 0.4655 | 0.4534 | 0.4579 | 0.4632 | 0.4519 | 0.4443 |
| control held-out, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control held-out, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |
| defective training score | n/a | 0.004 | 0.020 | 0.004 | 0.021 | 0.008 | 0.008 |
| control training score | n/a | 0.541 | 0.559 | 0.647 | 0.580 | 0.658 | 0.693 |

Wrong diagnoses: data too hard or model too small (44% of its rollouts are right), a broken reward or
extraction (unchanged, and it scores the test split correctly), the answer format, lr/KL/GRPO
settings, dataset size (same row count).

## Reference solution

- **issue**: the one-sided filter → `zip` pairs questions with later answers → correct responses
  score 0 → no learning signal. "Labels are wrong" without the filter earns partial credit.
- **solution**: filter both lists with the same index (`fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. The grader also checks that every regenerated training row carries its own gold answer. Defective step 60: 0.4443 (best 0.4655); control 0.5580 / 0.5406 (seeds 42 / 43);
  margins 0.044 / 0.031. The control's step 0 is 0.4503, so a run that does not learn fails. One
  defective run and two control runs: margins, not a variance estimate. The regenerated training parquet
  is checked against the pinned raw split and `metrics.jsonl` must cover steps 0-60. Grader replay on the
  recorded evidence: defective exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest). The defective run used an
A100 80 GB; control seed 42 an A100 40 GB, seed 43 an A100 80 GB.
