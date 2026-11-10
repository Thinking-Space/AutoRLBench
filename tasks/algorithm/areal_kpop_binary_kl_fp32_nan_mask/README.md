# KPop binary-KL filter masks every saturated token in float32

Training Qwen1.5-MoE-A2.7B-Chat on GSM8K with asynchronous decoupled PPO and KPop token masking in AReaL. The binary-KL metric clamps probabilities to `1.0 - 1e-8`, which is exactly `1.0` in float32, so every sampled token of probability 1.0 gets a NaN metric and is masked: 0.0829 of tokens at step 0, rising to 0.449. Held-out GSM8K stays flat (0.6073 → 0.6021) and greedy decoding of the final policy collapses (GSM8K 0.0205).

| | |
| --- | --- |
| **Author** | Kaixiang Zhao |
| **Affiliation** | Brigham Young University |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `float32-precision` `nan-metric` `spurious-token-masking` `importance-weight-filter` `silent-failure` |
| **Framework** | AReaL v2.0.0 (+33 commits, `9f2967a4`), decoupled PPO with KPop masking, SGLang 0.5.10.post1 rollouts, FSDP actor |
| **Run time** | ~1.2 hours (87 steps) |
| **Resources** | 8 × A100-SXM4 (80 GB) |

## Difficulty

The defect is in upstream AReaL, `areal/utils/functional/functional.py::compute_binary_kl_divergence`:

```python
p = torch.clamp(torch.exp(log_p), eps, 1.0 - eps)   # eps = 1e-8, float32
q = torch.clamp(torch.exp(log_q), eps, 1.0 - eps)
return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
```

In float32, `1.0 - 1e-8 == 1.0`, so the upper clamp is a no-op. Any sampled token with probability exactly 1.0 on either side makes the second term `0 * log(0)` (NaN) or `x * log(x / 0)` (inf). `torch.maximum` keeps the NaN, and `_check_bounds` tests `metric <= upper`, which is False for both. KPop therefore drops every saturated token from the loss, independent of how well the trainer and the inference engine agree on it.

Nothing fails loudly:

- the run completes all 87 steps, and every logged statistic is finite;
- entropy stays at 0.18 to 0.22 and response length falls from 294 to 202 tokens;
- training reward rises from 0.718 to 0.875;
- the bound in the config (`upper: 2.0`) is loose, which points suspicion away from the filter.

The tell is that the filter is busy when it should be idle. At step 0 the policy has not moved and the importance weights sit at 1, yet `rs_filtered_fraction` is 0.0829. It then grows as the policy sharpens, because more sampled tokens reach probability 1.0 in float32. Real mismatch would not grow that way. Steps count from 0 for the first training step (`Train step 1/87` in the log).

| step | 0 | 20 | 40 | 77 | 86 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rs_filtered_fraction`, shipped float32 code | 0.0829 | 0.2433 | 0.3438 | 0.4487 (peak) | 0.4287 |
| `rs_filtered_fraction`, float64 binary KL (control) | 0.0001 | 0.0059 | 0.0196 | 0.0271 | 0.0208 |

The frozen per-token dumps (`run/ktdump/seeds_kpop_s2/`, indices 0, 680 and 1340) hold the inference-engine and proximal logprobs, so the bug reproduces offline on the run's own tokens:

- at step 0 the shipped function masks 5.33% of dumped tokens, 5.32% of them through a NaN metric, against 0.008% in float64;
- 99.8% of the masked tokens have probability exactly 1.0 on at least one side, with a mean sampled probability of 0.9997;
- later dumps mask 33.1% of tokens, against 2.0% and 1.8% in float64.

Held-out GSM8K (`eval-rollout/reward`) is 0.6073 / 0.6128 / 0.6021 after epochs 1 to 3, against 0.6238 / 0.6541 / 0.6518 for the same seed with a float64 binary KL. During the final evaluation, five rollouts fail with SGLang's "All output_tokens are EOS or PAD tokens", meaning the policy emitted an empty response. Greedy pass@1 of the final checkpoint (`run/eval/final_checkpoint_suite.tsv`) is GSM8K 0.0205, MATH500 0.0260 and SVAMP 0.0233, against 0.5550 / 0.2120 / 0.5900 for the base model and 0.6050–0.6603 / 0.2220–0.2460 / 0.6133–0.7200 for the three float64 controls. The greedy outputs themselves were not saved, so the decoding failure mode is not characterized beyond the empty-response errors.

Several diagnoses are plausible but wrong:

- "the train-inference mismatch of the MoE model is large", since the same tokens pass in float64;
- "the KPop bound is too tight";
- asynchronous off-policyness, the learning rate, reward scaling, the reward or the data, since all are identical in the controls;
- "training did not improve", since training reward and epoch-1 held-out reward both rise.

## Reference solution

- **issue**: name the float32 clamp in `compute_binary_kl_divergence` (`1.0 - 1e-8` is `1.0`, so saturated tokens give NaN/inf) and its path through `torch.maximum` and `_check_bounds` to the masked fraction, which is far above zero at step 0 and grows with confidence. "KPop masks too many tokens" or "the bound is too tight" without the numerical cause earns partial credit. Train-inference mismatch, hyperparameters and "no learning" fail.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.

Issue-only: the in-run evaluation is logged once per epoch with no step-0 point. The second seed with the shipped code (final `eval-rollout/reward` 0.6533, greedy GSM8K 0.5732) overlaps the controls on that metric, so no single threshold separates the arms. The spurious masking itself (0.0810 at step 0, peak 0.314 for that seed) is present in every run with the shipped code.

Recorded runs, all on 8 × A100 with an identical resolved `config.yaml`:

- Shipped float32 code, two seeds. Seed 2 is shipped in `workspace/`.
- Float64 binary KL (eps 1e-7, NaN from saturation on both sides mapped to 0), three seeds. Final `eval-rollout/reward` 0.6311 / 0.6518 / 0.6507; masked fraction at most 0.033.
