# GRPO policy ratio ignores rollout log probabilities

Training Qwen/Qwen2.5-0.5B-Instruct with GRPO on official GSM8K. The policy
loss uses current log probabilities in place of frozen rollout probabilities,
forcing the probability ratio to one and disabling clipping.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | algorithm |
| **Tags** | `verl` `grpo` `fsdp` `vllm` `gsm8k` |
| **Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Dataset** | openai/gsm8k main; first 256 train rows, 100 test questions sampled with seed 42 |
| **Framework** | verl 0.3.0.post1, FSDP, `scripts/train.sh` |
| **Environment** | `verl_policy_ratio_recomputes_old_logprob` · 1 × A800 80 GB GPU · 40 GiB container RAM |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, 30 steps each |

## Difficulty

In `verl/trainer/ppo/core_algos.py`, subtracting detached current log
probabilities makes the ratio one throughout four actor passes over each
rollout. The resulting loss diverges from the frozen behavior-policy objective.

| Measurement | Defective | Repaired |
| --- | ---: | ---: |
| Initial accuracy, 100 questions | 47% | 47% |
| Final accuracy, step 30 | 0% | 50% |
| Maximum behavior-objective loss error | 0.15077248 | 0 |

Each run uses batch size 16, group size 8 and four actor passes per step
(120 optimizer updates). Validation runs at steps 0, 10, 20 and 30 on the
same 100 questions. These results are from one pair of runs.

## Reference solution

- **issue**: identify the incorrect log-probability subtraction and explain
  the inactive clipping and behavior-objective mismatch.
- **solution**: use frozen rollout behavior log probabilities in the policy
  ratio and KL calculation and submit `report.md`; the harness then runs the
  repaired experiment for 30 steps. The reference
  patch and calibration evidence are hidden in the solution oracle.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted
  0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: complete the fixed 30-step training run with correct policy
  loss and KL, then achieve **greater than 45% accuracy** on all 100 validation
  questions at step 30 (at least 46 correct). The verifier independently
  scores the final answers and checks the training artifacts.
  Score: 1.0 if all checks pass, otherwise 0.0.
