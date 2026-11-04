# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

The training data is too hard for this model, so GRPO receives almost no
learning signal. Both parts are essential:

1. **Data difficulty.** `workspace/taco_verl/train.parquet` contains only TACO
   `VERY_HARD` problems (2329 rows), while `workspace/taco_verl/test.parquet`
   is the 184-row TACO `EASY` validation set. The training set is far beyond
   what Qwen2.5-1.5B-Instruct can solve.
2. **Causal chain to the failure.** The TACO reward is binary (1 only when every
   test passes). On the training problems the policy almost never earns a 1,
   so nearly every GRPO group of 8 rollouts is all-zero, its group-normalized
   advantage is exactly 0, and the policy-gradient term vanishes on most steps.
   Held-out accuracy fluctuates without a clear improvement over its step-0
   level.

The train/validation difficulty mismatch is how the agent can tell the
training set is too hard (same model, same scorer, roughly 10x higher pass rate
on the validation problems). The mismatch by itself is not the failure
mechanism: the run would stall the same way with a `VERY_HARD` validation set.

## Observable impact

From `workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261021_080557.log`
(400/400 steps, 145 steps per epoch) and the per-step rollout dumps in
`workspace/rollout_data/` (128 rollouts per step = 16 prompts x 8):

- `critic/rewards/mean` averages 0.0078 over 400 steps; 254 steps are exactly
  0; the maximum is 0.1016.
- 256 of 400 steps have `critic/advantages/max` = `critic/advantages/min` = 0
  and `actor/pg_loss` = 0. `actor/grad_norm` on those steps is 0.0037-0.020
  (median 0.0054): only the KL and entropy terms remain.
- Across 51,200 training rollouts the success rate is 0.78%. Only 175 of 6,400
  GRPO groups (2.7%) contain both a correct and an incorrect rollout; 2 groups
  are all-correct; the rest are all-zero. Of 2,151 distinct training prompts,
  104 are ever solved.
- Same model, same scorer: validation `val-aux/taco/pass_rate/mean@1` is 0.2246
  at step 0, while the training per-test pass rate over the run averages about
  0.026.
- Held-out `val-core/taco/acc/mean@1` (184 rows): 0.0707 (13/184) at step 0,
  0.0924 at step 50, 0.0870 at steps 100 and 150, 0.0707 at step 200, 0.0815 at
  step 250, 0.0707 at step 300, 0.0652 at step 350, and 0.0761 (14/184) at
  step 400. The curve is flat within binomial noise (standard error about
  0.019): 400 steps of training produce no held-out improvement.

## Symptoms, not root causes

- `actor/entropy` rises from 0.69 to about 1.0-1.1 (peak 2.04). On zero-advantage
  steps the only loss terms left are the entropy bonus and the KL penalty, so
  the policy drifts slightly. This does not cause the failure; held-out
  accuracy does not degrade.
- `actor/kl_loss` grows slowly from 0 to about 0.005.
- More training responses fail to parse (incomplete or Python 2 code) as
  entropy rises.

## Not the root cause

- KL / entropy settings (`kl_loss_coef=0.02`, `entropy_coeff=0.001`), the
  learning rate (`1e-6`), and batch shape (16 prompts x 8 rollouts) are ordinary
  values. A larger batch or a different coefficient does not create a learning
  signal when 97% of groups are all-zero.
- Response length: mean length stays between about 300 and 750 tokens, and the
  clip ratio averages 0.2%.
- Reward or code defects: `verl/utils/reward_score/taco.py`, the launch script,
  the model, and `verl/utils/torch_functional.py` are the healthy baseline.
  The same scorer gives a validation pass rate of about 0.22. The binary
  all-tests-pass reward amplifies the sparsity but is the standard TACO reward,
  not a defect.
- Reward hacking: hard-coded outputs such as `print(<constant>)` are 9 of 51,200
  training rollouts and 0 of 1,656 validation rollouts.
- Evaluation artifact: the 184-row validation protocol is consistent across all
  nine evaluations; the problem is that training does not learn, not that the
  measurement is unreliable.

## Evidence

- The parquet rows do not carry an `extra_info.difficulty` label. Difficulty can
  be established from the problem statements and from the train/validation
  pass-rate gap under the same model and scorer. Citing a `difficulty` field is
  not required.

Audit note (2026-10-31): all 51,200 training records parse when read as
LF-delimited JSONL. One response in `workspace/rollout_data/380.jsonl` contains
a legal Unicode U+2028 character; `str.splitlines()` incorrectly splits that
record. Group by prompt, not adjacent blocks of eight: the intact evidence
has 6,223 all-zero, 175 mixed and 2 all-correct groups, each with eight samples.
Exact `VERY_HARD`/`EASY` labels are not required for a diagnosis that establishes
excessive difficulty and the GRPO signal chain.
