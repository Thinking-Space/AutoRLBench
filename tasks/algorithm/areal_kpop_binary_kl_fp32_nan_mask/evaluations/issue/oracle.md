# Canonical issue oracle

Reconciled from the task's frozen evidence and from recorded control runs. The controls
have the same resolved configuration (`config.yaml`) and differ only in the binary-KL
computation (float64, with an exactly representable clamp).

Expected verdict: **unhealthy**.

## Root cause

- `areal/utils/functional/functional.py::compute_binary_kl_divergence`, as used by
  `apply_rejection_sampling` when `rejection_sampling.metric == "binary_kl"` (KPop)
  - kind: `symbol`
  - observed: the probabilities are clamped in float32 with
    `torch.clamp(torch.exp(log_p), eps, 1.0 - eps)` and `eps = 1e-8`. In float32,
    `1.0 - 1e-8` rounds to exactly `1.0`, so the upper clamp does nothing. A sampled token
    whose probability is exactly 1.0 on either side (logprob 0 in float32) gives
    `1 - p = 0` or `1 - q = 0`. The second term
    `(1 - p) * torch.log((1 - p) / (1 - q))` then evaluates to `0 * log(0) = NaN` or `inf`.
    `torch.maximum(kl_fwd, kl_rev)` propagates the NaN, and `_check_bounds` tests
    `metric <= config.upper`, which is False for NaN and inf. Every saturated token is
    therefore classified as out of bounds and dropped from the policy loss, whatever its
    real train/inference agreement.
  - expected: an eps that is representable next to 1.0 in the working dtype, or the
    computation in float64, with saturated-on-both-sides tokens mapped to KL 0. With that,
    the filter drops almost nothing on a run whose importance weights sit near 1.

A passing diagnosis names the float32 clamp / NaN in `compute_binary_kl_divergence` and
traces it through the bounds check to the masked fraction. The defect is not the threshold:
`upper: 2.0` is a very loose binary-KL bound. A diagnosis that blames the KPop threshold, or
says only that "KPop masks too many tokens", earns partial root-cause credit at most.

## Observable impact

- Step numbers below count from 0 for the first training step, which the log prints as
  `Train step 1/87`.
- Primary source evaluation metric: `eval-rollout/reward` (GSM8K test, end of each epoch,
  in `run/experiments/logs/user/kt-seeds/kpop-s2/main.log`): 0.6073, 0.6128, 0.6021 after
  epochs 1, 2 and 3. Held-out accuracy does not improve after the first epoch.
- `ppo_actor/update/rs_filtered_fraction` (masked token fraction):
  - 0.0829 at step 0, when the policy has not moved yet;
  - 0.2433 at step 20, 0.3438 at step 40, peak 0.449 at step 77, 0.429 at the final step 86.
  - It exceeds 0.10 from step 7 on.
- At step 0 the importance weights are close to 1, so a working filter with a 2-nat bound
  should mask essentially nothing. The fraction also grows as the policy sharpens, which
  is the signature of saturation rather than mismatch: more sampled tokens reach
  probability 1.0 in float32.
- Entropy (0.183 at step 0, at most 0.22) and length (294 to 202 tokens) look healthy.
  Training reward rises from 0.718 to 0.875 and ends at 0.824. Nothing fails loudly.
- The per-token dumps in `run/ktdump/seeds_kpop_s2/` (indices 0, 680, 1340, i.e. steps 0,
  38, 84) hold the inference-engine and proximal logprobs. Re-running the shipped function
  on them reproduces the defect:
  - Step 0: 5.33% of dumped tokens are masked and 5.32% have a NaN metric. A float64
    computation masks 0.008%.
  - Step 0: 99.8% of the masked tokens have probability exactly 1.0 on at least one side,
    and their mean sampled probability is 0.9997, against 0.904 for all tokens.
  - Steps 38 and 84: 33.1% of dumped tokens are masked (31.1% and 31.3% NaN), against
    2.0% and 1.8% in float64.
- Final-checkpoint evaluation (`run/eval/final_checkpoint_suite.tsv`, greedy pass@1)
  against the untrained base model:
  - GSM8K 0.0205 (base 0.5550), MATH500 0.0260 (0.2120), SVAMP 0.0233 (0.5900).
  - Greedy decoding of the final policy has collapsed, although sampled held-out reward is
    still 0.60.
- During the epoch-3 held-out evaluation (logged between `Train step 86/87` and
  `Train step 87/87`), five rollouts fail
  in `rollout.log` / `eval-rollout.log`: SGLang returns 500 with "All output_tokens are EOS or
  PAD tokens", meaning the policy emitted an empty response. The failures are non-fatal and do
  not occur in any control run.

## Trigger conditions

- `rejection_sampling.metric: binary_kl` (KPop) with float32 logprobs. Sampled tokens of
  probability exactly 1.0 exist from step 0 on any instruction-tuned model, and there are
  more of them as the policy becomes confident.
- In the recorded control runs (same resolved configuration, float64 binary KL):
  - the masked fraction is 0.0000 to 0.0001 at step 0 and at most 0.033 over all 87 steps;
  - `eval-rollout/reward` rises to 0.6311 / 0.6518 / 0.6507 (seeds 1/2/3);
  - final greedy GSM8K is 0.6050 / 0.6603 / 0.6490.
- A second seed with the shipped code masks 0.0810 at step 0, peaks at 0.314, and ends with
  greedy GSM8K 0.5732.

## Documentation reconciliation

- The flat held-out curve and the collapsed greedy accuracy are downstream of the defect;
  they are not separate root causes. The second shipped-code seed is less affected
  (final `eval-rollout/reward` 0.6533), so a single seed's outcome varies. The 8% to 45%
  spurious masking is present in every run with the shipped code.
- The run is not "not learning". Training reward rises to 0.875 and epoch-1 held-out reward
  is 0.6073. The defect is a silently wrong filter that removes a large and growing share of
  the most confident tokens, relative to what the same recipe does with a correct
  binary-KL. Saying "training did not improve" is a wrong attribution.
- These are not the root cause, because each is identical in the controls that stay healthy:
  - a genuine train-inference mismatch of the MoE model;
  - the KPop bound `upper: 2.0`;
  - asynchronous off-policyness (`max_head_offpolicyness: 2`);
  - the learning rate (3e-6);
  - `reward_scaling: 10`;
  - the clip range;
  - the reward function or the data;
  - the weight sync.
- There is no OOM and no crash, and every logged statistic is finite: the NaNs live only
  inside the filter.
