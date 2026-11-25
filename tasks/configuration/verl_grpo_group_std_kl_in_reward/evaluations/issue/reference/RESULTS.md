# Recorded comparison

Five 60-step runs of `run_grpo_gsm8k.sh`, all with the same image, data, model, budget and code.
The defective arm is the shipped workspace. The control is the same workspace with `fix.diff` applied:
one more line in the launch script, `algorithm.norm_adv_by_std_in_grpo=False`. Each arm ran once per seed (42, 43).
The shipped evidence is the seed-42 defective run (`workspace/results/grpo_gsm8k_seed42`).
A third arm, plain GRPO, is the same workspace with `algorithm.use_kl_in_reward=False` (seed 42 only). `metrics.jsonl` of the other 4 runs sits next to this file.

Held-out accuracy is measured on the 1,319 GSM8K test questions with greedy decoding, from
`val_generations/<step>.jsonl`. Each cell gives two numbers:
1. the reward's `acc`, which is what `val-core/openai/gsm8k/acc/mean@1` logs (verl's flexible
   extraction, string match against the gold);
2. a numeric re-score: the last number in the response must equal the gold numerically, so "18.00"
   counts for "18".

| held-out accuracy | step 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4299 / 0.4587 | 0.4556 / 0.4769 | 0.4647 / 0.4784 | 0.4913 / 0.5027 | 0.4852 / 0.4913 | 0.4875 / 0.4996 | 0.4913 / 0.4951 |
| defective, seed 43 | 0.4284 / 0.4572 | 0.4443 / 0.4670 | 0.4723 / 0.4860 | 0.4814 / 0.4913 | 0.4958 / 0.5034 | 0.4890 / 0.4943 | 0.4943 / 0.4996 |
| control, seed 42 | 0.4299 / 0.4602 | 0.4814 / 0.4905 | 0.5080 / 0.5087 | 0.5307 / 0.5307 | 0.5459 / 0.5459 | 0.5337 / 0.5337 | 0.5534 / 0.5534 |
| control, seed 43 | 0.4268 / 0.4556 | 0.4602 / 0.4708 | 0.5034 / 0.5080 | 0.5118 / 0.5118 | 0.5133 / 0.5140 | 0.5201 / 0.5216 | 0.5398 / 0.5398 |
| plain GRPO, no KL term, seed 42 | 0.4412 / 0.4738 | 0.4716 / 0.4799 | 0.4882 / 0.4890 | 0.5193 / 0.5193 | 0.5262 / 0.5262 | 0.5254 / 0.5254 | 0.5186 / 0.5186 |

Step-0 `acc` differs by up to 0.0144 and the numeric re-score by up to 0.0182 between
identical starts: vLLM's batched greedy decoding is not bit-deterministic.

Training metrics, steps 1-60:

| metric | defective, seed 42 (shipped) | defective, seed 43 | control, seed 42 | control, seed 43 | plain GRPO, no KL term, seed 42 |
| --- | --- | --- | --- | --- | --- |
| `critic/score/mean`, step 1 | 0.326 | 0.373 | 0.312 | 0.340 | 0.270 |
| `critic/score/mean`, mean of steps 51-60 | 0.480 | 0.508 | 0.661 | 0.670 | 0.647 |
| `actor/reward_kl_penalty`, steps 10 / 30 / 60 | 0.0028 / 0.0067 / 0.0105 | 0.0023 / 0.0057 / 0.0097 | 0.0206 / 0.0702 / 0.1002 | 0.0209 / 0.0613 / 0.0998 | not logged / not logged / not logged |
| mean KL per response at step 60 (nats) | 3.14 | 3.03 | 28.42 | 27.79 | not logged |
| largest `score − rewards` gap | 0.0030 | 0.0031 | 0.0286 | 0.0266 | 0.0000 |
| `actor/entropy`, step 60 | 0.344 | 0.406 | 0.093 | 0.126 | 0.076 |
| `response_length/mean`, step 1 → 60 (tokens) | 322 → 299 | 294 → 313 | 325 → 284 | 295 → 278 | 335 → 276 |
| `actor/grad_norm`, range | 0.64-0.90 | 0.75-0.92 | 0.26-0.40 | 0.24-0.42 | 0.63-1.02 |
| groups per step with no accuracy variation, range over 60 steps | 11-29 of 64 | 12-25 of 64 | 13-36 of 64 | 14-37 of 64 | 16-37 of 64 |

## The mechanism

The rollout dumps do not record each sample's KL term, so the per-sample advantage cannot be replayed
from them. `group_std_demo.py` shows the arithmetic with verl's own `compute_grpo_outcome_advantage` on
one synthetic group of 8 wrong answers whose summed log-ratios span a few nats (output in
`group_std_demo.txt`): with the default flag the advantages are of order 1, with
`norm_adv_by_std_in_grpo=False` they are of order 0.001. The recorded control keeps the same penalty and
coefficient without the division; it also rescales the groups in which accuracy varies, so it is not a
pure isolation. The plain-GRPO arm (std on, `use_kl_in_reward=False`, seed 42; `plain_grpo_seed42/metrics.jsonl`) keeps the division and drops the KL term; it learns too (0.4412 → 0.5186). Between them, the two reference arms show that only the combination stalls.

## Why issue-only

Over steps 40-60 the defective runs read 0.4852-0.4958 and the controls 0.5133-0.5534. That is from
two seeds per arm, one run each; no rerun threshold was calibrated on it.
