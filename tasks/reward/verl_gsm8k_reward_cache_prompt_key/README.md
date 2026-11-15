# Reward memo keyed by prompt: identical scores per GRPO group

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, the reward function memoizes scores by prompt index, so all eight rollouts of a prompt get the first rollout's score. Every GRPO advantage is 0, the policy never moves, and held-out accuracy stays at 0.4617 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | NTU |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | `reward-cache` `zero-advantage` `silent-failure` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~35 minutes |
| **Resources** | 1 × A100 (40 GB) |

## Difficulty

`reward/gsm8k_reward.py`:

```python
_SCORE_CACHE: dict = {}

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    key = (data_source, extra_info.get("split"), extra_info.get("index"))
    if key not in _SCORE_CACHE:
        _SCORE_CACHE[key] = gsm8k.compute_score(solution_str, ground_truth, method="flexible")
    return _SCORE_CACHE[key]
```

The key omits the response. `extra_info.index` is the prompt's row index, shared by the eight
rollouts of a group and by every re-evaluation of a test prompt, and the single reward worker lives
for the whole run.

`actor/pg_loss`, `actor/grad_norm` and `actor/kl_loss` are 0.0 at all 60 steps; entropy stays at
0.49-0.53 (control → 0.08). The training score moves between 0.27 and 0.41, which looks like noisy
learning. The logged validation accuracy is exactly 0.4337 at all seven evaluations (the cached
step-0 scores), while re-scoring the fresh generations gives 0.453-0.466. In the rollouts at steps 1,
30 and 60, all 192 groups carry one score, and 473 of 1,536 scores disagree with the response's own
final answer.

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective logged val acc | 0.4337 | 0.4337 | 0.4337 | 0.4337 | 0.4337 | 0.4337 | 0.4337 |
| defective held-out | 0.4655 | 0.4632 | 0.4541 | 0.4625 | 0.4534 | 0.4617 | 0.4617 |
| control held-out, seed 42 | 0.4503 | 0.4996 | 0.4973 | 0.5254 | 0.5413 | 0.5375 | 0.5580 |
| control held-out, seed 43 | 0.4602 | 0.4905 | 0.5027 | 0.5353 | 0.5239 | 0.5557 | 0.5406 |

Wrong diagnoses: healthy (validation is perfectly stable), lr or KL (the gradient is exactly 0), a
GRPO grouping or uid bug (that code is unchanged), the scorer (unchanged).

## Reference solution

- **issue**: the cache key omits the response → identical scores within each group → zero advantage;
  the logged validation replays step 0. "Zero gradient" without the cache earns partial credit.
- **solution**: remove the memo (`fix.diff`); keying it on the response is equivalent.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. Defective step 60: 0.4617 (best 0.4655); control 0.5580 / 0.5406 (seeds 42 / 43);
  margins 0.044 / 0.031. The control's step 0 is 0.4503, so a run that does not learn fails. One
  defective run and two control runs: margins, not a variance estimate. The regenerated training parquet
  is checked against the pinned raw split and `metrics.jsonl` must cover steps 0-60. Grader replay on the
  recorded evidence: defective exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the NVIDIA CDI on the
same image digest).
