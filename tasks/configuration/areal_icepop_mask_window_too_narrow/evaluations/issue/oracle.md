# Canonical issue oracle

Reconciled from the task's frozen evidence and from the recorded control runs, which share
the resolved configuration (`config.yaml`) except for the IcePop window.

Expected verdict: **unhealthy**.

## Root cause

- `run.sh`: `actor.rejection_sampling.lower=0.67` / `actor.rejection_sampling.upper=1.5`
  - kind: `hparam`
  - observed: the IcePop token mask (`rejection_sampling`: `level: token`, `action: mask`,
    `metric: ratio`) keeps only tokens whose behaviour importance weight
    `exp(prox_logp - old_logp)` lies in [0.67, 1.5]; every other token is dropped from
    the policy loss.
  - expected: the window that `examples/math/gsm8k_icepop.yaml` ships, [0.5, 5].

The window is too narrow for this run. Here the ratio carries both the FSDP-versus-SGLang
logprob gap of the MoE model and up to one policy version of asynchronous drift
(`max_head_offpolicyness: 2`, proximal staleness 1.0). The narrow window drops a much larger
and growing share of sampled tokens, concentrated on low-probability tokens. About 42 steps
in, the masked share stops being bounded and runs away together with entropy and response
length. The policy then degenerates and held-out accuracy falls. A passing diagnosis names
the narrowed window and connects it to the runaway masked fraction. Saying only that "too
many tokens are masked", or blaming the train-inference mismatch in general, without the
window override earns partial root-cause credit at most.

## Observable impact

- Primary source evaluation metric: `eval-rollout/reward` (GSM8K test, end of each epoch,
  in `run/experiments/logs/user/kt-seeds/icepop_w067-s2/main.log`): 0.6448, 0.5463, 0.5777
  after epochs 1, 2 and 3. Accuracy rises in epoch 1, then falls.
- `ppo_actor/update/rs_filtered_fraction` (masked token fraction): 0.0161 at step 0, 0.0493
  at step 20, 0.0737 at step 40. It first exceeds 0.10 at step 43, peaks at 0.750 at step
  73, and is 0.672 at the final step 86.
- `ppo_actor/update/entropy/avg`: 0.184 at step 0, peak 5.92 at step 65. `ppo_actor/seq_len/avg`:
  284 at step 0, peak 600 at step 68, 489 at step 86. `ppo_actor/task_reward/avg`: 0.736 at
  step 0, at most 0.807, 0.702 at step 86. `ppo_actor/no_eos_ratios/avg` stays at or below
  0.0195, so the longer responses are not truncation.
- The per-token dumps in `run/ktdump/seeds_icepop_w067_s2/` (indices 0, 1180 and 2340) hold
  the inference-engine and proximal logprobs, so the window can be evaluated
  counterfactually on the same tokens:
  - Step 0 (index 0): 1.81% of tokens fall outside [0.67, 1.5], against 0.51% outside the
    shipped [0.5, 5].
  - Sampled tokens with probability below 0.5 make up 8.8% of tokens but 83.5% of the
    masked ones (17.1% of them are masked, against 0.33% of the rest).
  - In the step-52 dump (index 1180), 48.4% of the dumped tokens are masked, and their mean
    sampled-token probability has fallen from 0.898 to 0.544.
- Final-checkpoint evaluation (`run/eval/final_checkpoint_suite.tsv`, greedy pass@1) against
  the untrained base model:
  - GSM8K is 0.6073 (base 0.5550).
  - MATH500, Minerva and Olympiad fall below the base model: 0.1360 / 0.0221 / 0.0208,
    against 0.2120 / 0.0404 / 0.0490.

## Trigger conditions

- Token-level IcePop masking on the decoupled behaviour importance weight, with a window
  narrower than the natural spread of that weight. That spread is set by the MoE
  train-inference logprob gap plus asynchronous staleness, and it is widest on sampled
  low-probability tokens.
- Any seed. In the recorded control runs (same resolved configuration, window [0.5, 5]), the
  masked fraction stays at or below 0.046 for all 87 steps, entropy never exceeds 0.23, and
  `seq_len/avg` falls to about 220. `eval-rollout/reward` there is 0.627 / 0.652 / 0.653
  (seed 2) and 0.625 / 0.653 / 0.669 (seed 3). Two more seeds with the narrow window
  reproduce the runaway: the masked fraction crosses 0.10 at steps 43 and 42, entropy peaks
  at 7.53 and 2.65, and the final `eval-rollout/reward` is 0.380 and 0.507.

## Documentation reconciliation

- The rising entropy, the length growth, the late fall in training reward and the fall in
  held-out accuracy are symptoms of the runaway masking, not separate root causes.
- The run is not "not learning". Training reward peaks at 0.807 and held-out GSM8K rises in
  epoch 1 (0.6448), and final GSM8K is above the base model. The defect is a degeneration
  relative to what the same recipe reaches with the shipped window. Saying "training did not
  improve" is a wrong attribution.
- These are not the root cause, because each is identical in the control runs that stay
  healthy:
  - the MoE model or a train-inference mismatch in general;
  - asynchronous off-policyness (`max_head_offpolicyness: 2`);
  - the learning rate (3e-6);
  - `reward_scaling: 10`;
  - the clip range (0.2 / 0.28);
  - the 1024-token generation limit;
  - the reward function, the data, or the SGLang/xccl weight sync.
- No worker error or OOM occurs in this run. Infrastructure is not the cause.
