# Canonical issue oracle

Reconciled from the task's frozen evidence and the recorded control. The control is the same tree,
data, seed, budget and image; its launch script has one more line,
`algorithm.norm_adv_by_std_in_grpo=False`. No framework code differs between the arms.

Expected verdict: **unhealthy**.

## Root cause

- `run_grpo_gsm8k.sh`: `algorithm.use_kl_in_reward=True` (with `kl_penalty=kl`, `kl_ctrl.type=fixed`,
  `kl_ctrl.kl_coef=0.001`) together with GRPO's `algorithm.norm_adv_by_std_in_grpo=True` (verl's
  default; the script does not set it)
  - kind: `config` (two settings interacting with framework behaviour)
  - observed: `verl/trainer/ppo/ray_trainer.py::apply_kl_penalty` subtracts
    `0.001 × (log π − log π_ref)` from the reward of every response token. GRPO's
    `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` sums the token rewards per sample and
    standardises them inside each prompt group of 8 samples: `(score − group mean) / (group std + 1e-6)`.
    In a group whose 8 answers are all right or all wrong, the 0/1 part is constant, so the only
    variation is 0.001 times each sample's summed log-ratio to the reference model, and the division
    scales it to unit variance. Such a group gets advantages as large as a group in which accuracy
    varies, and they favour the samples whose probability the policy has raised the least: in
    expectation a unit-scale push toward the initial model. The 0.001 coefficient cancels there; any
    coefficient whose term varies by well over the 1e-6 epsilon gives the same advantages (from step 2
    on; at step 1 the log-ratio is 0). In the kept rollout dumps 13-27 of the 64 groups per step
    are of this kind. The policy is therefore held close to the initial model and GRPO improves
    it only a little.
  - expected: the KL penalty stays at its nominal size relative to the 0/1 reward, as in the control:
    mean-centred advantages (`algorithm.norm_adv_by_std_in_grpo=False`), or the KL term applied in the
    loss instead of the reward (`algorithm.use_kl_in_reward=False`, `actor.use_kl_loss=True`).

Naming the KL penalty in the reward **and** GRPO's per-group division by the standard deviation, with
its causal role (in groups without accuracy variation the penalty is the only signal and is scaled to
unit scale → those groups pull the policy back to the initial model → the policy barely moves → held-out
accuracy stalls) is a complete attribution. Do not require additional causes. Equivalent wording counts:
"advantage normalisation inside each group", "per-prompt whitening", "std-normalised advantages amplify
the KL term when all samples are equally correct". The count of such groups is welcome but not required.

**The per-group standardisation is required.** "The KL penalty in the reward keeps the policy near the
reference", "use_kl_in_reward should be off" or "the KL regularisation is too strong" without the group
standardisation earns at most partial root-cause credit: it names the switch but not why a term that
changes the reward by at most 0.0030 can hold back a 0/1 reward. "The coefficient 0.001 is too
large" is wrong and gets at most partial credit: the coefficient cancels in the groups that matter, and
the control keeps the same coefficient and learns. Blaming the learning rate, the step budget, PPO
clipping, the reward function, the answer extraction, the data, or calling the run healthy or merely
slow is wrong and gets no root-cause credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/grpo_gsm8k_seed42` plus the code and
configuration).

- Held-out accuracy gains little and stalls: logged `val-core/openai/gsm8k/acc/mean@1` reads
  0.4299 (step 0) → 0.4556 (10) → 0.4647 (20) → 0.4913 (30) → 0.4852 (40) → 0.4875 (50) → 0.4913 (60).
- The training score stops early: `critic/score/mean` is 0.326 at step 1, averages 0.464 over steps
  11-60 and 0.480 over steps 51-60; over steps 31-60 it gains only 0.0052 per 10 steps.
- The policy stays near the initial model: `actor/reward_kl_penalty` is 0.0028 at step 10 and
  0.0105 at step 60, `actor/entropy` is 0.536 at step 1 and still 0.344 at step 60, and
  `response_length/mean` goes 322 → 299 tokens.
  That is 3.1 nats of summed log-ratio per response at step 60
  (`actor/reward_kl_penalty` × `response_length/mean`).
- The run is unhealthy although held-out accuracy rises by 0.061. The main reason is that the
  regularisation actually applied is not the one configured: the KL term is at most 0.0030 of the
  reward, yet it is the entire advantage for 20-42% of the samples per step (kept dumps), at the
  same scale as the accuracy signal elsewhere, and the policy ends only 3.1 nats per response from the
  initial model. The flat training score (+0.0052 per 10 steps over 31-60) and held-out accuracy
  (within 0.4852-0.4913 from step 30) are consistent with that, but on their own they would not
  prove a defect. A diagnosis that finds the mechanism but calls it a harmless inefficiency and the
  run healthy has the wrong verdict.

Supporting detail, not a separate evidence item: the penalty is negligible in the logged reward
(`critic/score/mean − critic/rewards/mean` is at most 0.0030); `actor/grad_norm` stays at 0.64-0.90
and `actor/pg_clipfrac` at 0.0007-0.0014; nothing errors. In the kept rollout dumps 13-27 of the 64 prompt
groups have 8 answers that are all right or all wrong.

## Trigger conditions

- A 0/1 outcome reward, so that many prompt groups have no variation in the outcome.
- A per-sample term added to that reward before GRPO's group normalisation. Here it is the KL penalty
  that `use_kl_in_reward=True` puts into the token-level rewards.
- GRPO with `norm_adv_by_std_in_grpo=True`, which is verl's default and is not set in the launch
  script.
