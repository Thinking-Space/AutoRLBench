# IcePop mask window narrower than the train-inference ratio spread

Training Qwen1.5-MoE-A2.7B-Chat on GSM8K with asynchronous decoupled PPO and IcePop token masking in AReaL. The launcher narrows the mask window from the shipped [0.5, 5] to [0.67, 1.5]. About 42 steps in, the masked token fraction runs away (0.0161 at step 0 to 0.750 at step 73), entropy and response length blow up, and held-out GSM8K falls from 0.6448 after epoch 1 to 0.5777 at the end, against 0.653 with the shipped window.

| | |
| --- | --- |
| **Author** | Kaixiang Zhao |
| **Affiliation** | Brigham Young University |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `token-masking` `importance-weight-window` `train-inference-mismatch` `entropy-blowup` `delayed-collapse` |
| **Framework** | AReaL v2.0.0 (+33 commits, `9f2967a4`), decoupled PPO with IcePop masking, SGLang 0.5.10.post1 rollouts, FSDP actor |
| **Run time** | ~1.6 hours (87 steps) |
| **Resources** | 8 × A100-SXM4 (80 GB) |

## Difficulty

The defect is two Hydra overrides at the end of `run.sh`:

```bash
    actor.rejection_sampling.lower=0.67 \
    actor.rejection_sampling.upper=1.5
```

They replace the window that `examples/math/gsm8k_icepop.yaml` ships (`lower: 0.5`, `upper: 5.0`). With `level: token`, `action: mask` and `metric: ratio`, the policy loss drops every token whose behaviour importance weight `exp(prox_logp - old_logp)` falls outside the window. In this run that weight carries two things. One is the logprob gap between the FSDP trainer and the SGLang engine on a MoE model. The other is up to one policy version of asynchronous drift (`max_head_offpolicyness: 2`, proximal staleness 1.0). Both spread widest on sampled low-probability tokens. Nothing fails loudly:

- the run completes all 87 steps with no worker error or OOM;
- training reward rises from 0.736 to 0.807;
- held-out accuracy after epoch 1 (0.6448) matches the control (0.627 and 0.625).

The failure starts in epoch 2. In all three recorded seeds, the masked fraction first exceeds 0.10 at step 42 or 43. It then feeds on itself alongside entropy and response length (`no_eos_ratios` stays at or below 0.0195, so this is not truncation):

| step | 0 | 20 | 40 | 43 | 52 | 65 | 73 | 86 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rs_filtered_fraction`, window [0.67, 1.5] | 0.0161 | 0.0493 | 0.0737 | 0.1075 | 0.3494 | 0.7025 | 0.7500 | 0.6724 |
| `rs_filtered_fraction`, window [0.5, 5] (control) | 0.0041 | 0.0323 | 0.0386 | 0.0409 | 0.0372 | 0.0426 | 0.0415 | 0.0459 |
| `entropy/avg`, window [0.67, 1.5] | 0.184 | 0.173 | 0.644 | 1.909 | 1.903 | 5.921 | 0.211 | 0.482 |
| `entropy/avg`, window [0.5, 5] (control) | 0.182 | 0.138 | 0.108 | 0.105 | 0.091 | 0.108 | 0.095 | 0.099 |
| `seq_len/avg`, window [0.67, 1.5] | 284 | 297 | 312 | 352 | 456 | 546 | 552 | 489 |
| `seq_len/avg`, window [0.5, 5] (control) | 291 | 263 | 232 | 226 | 226 | 222 | 218 | 219 |

Entropy peaks at 5.92 (step 65) against at most 0.23 in the control, and `seq_len/avg` peaks at 600 (step 68) while the control falls to 219.

The frozen per-token dumps (`run/ktdump/`, indices 0, 1180 and 2340) contain the inference-engine and proximal logprobs, so the window's effect can be checked on the run's own tokens:

- at step 0, 1.81% of tokens fall outside [0.67, 1.5], against 0.51% outside [0.5, 5];
- sampled tokens with probability below 0.5 make up 8.8% of tokens but 83.5% of the masked ones;
- in the step-52 dump (index 1180), 48.4% of the dumped tokens are masked and their mean sampled-token probability has fallen from 0.898 to 0.544.

The final checkpoint is still above the base model on GSM8K (greedy pass@1 0.6073 against 0.5550), so "the run did not learn" is wrong. On MATH500, Minerva and Olympiad it falls below the base model (0.1360 / 0.0221 / 0.0208, against 0.2120 / 0.0404 / 0.0490). The control's final checkpoints score 0.6634 and 0.6513 on GSM8K, and 0.2460 and 0.2520 on MATH500.

Several diagnoses are plausible but wrong, because each factor is identical in the healthy control:

- the MoE model or a train-inference mismatch in general;
- asynchronous off-policyness;
- the learning rate (3e-6);
- `reward_scaling: 10`;
- the clip range (0.2 / 0.28);
- the 1024-token generation limit;
- the reward function or the data;
- the xccl weight sync.

The control's resolved `config.yaml` differs from this run's only in `rejection_sampling.lower/upper`, plus an unused vLLM flag.

## Reference solution

- **issue**: name the narrowed IcePop window in `run.sh` ([0.67, 1.5] in place of the shipped [0.5, 5]) as the root cause, and connect it to the runaway `rs_filtered_fraction`: the masked share starts about 4× higher, then grows past 0.10 at step 43 and reaches 0.750 at step 73. Entropy blow-up, length growth and the epoch-2/3 accuracy drop are its symptoms. "Too many tokens are masked" or "train-inference mismatch" without the window override earns partial credit. "Training did not improve", the learning rate, off-policyness, reward scaling, the reward function and infrastructure all fail.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.

Issue-only: the in-run evaluation (`eval-rollout/reward`) is logged once per epoch with no step-0 point, so a solution threshold cannot be shown to clear the control's step-0 value. The rerun would also need the full 8-GPU MoE stack inside the image, which was not exercised in a container.

Recorded runs:

- Window [0.67, 1.5], three seeds on 8 × A100. Seed 2 is shipped in `workspace/`. Final `eval-rollout/reward` per seed: 0.380 (seed 1), 0.5777 (seed 2), 0.507 (seed 3).
- Window [0.5, 5], two seeds with an identical resolved configuration. Final `eval-rollout/reward`: 0.653 (seed 2), 0.669 (seed 3).

Seeds 1 and 3 of the narrow arm ran with SGLang memory caps that only affect scheduling (`mem_fraction_static=0.6`, plus `max_prefill_tokens=8192` for seed 1), after long degenerate prefills ran out of memory. Seed 2 ran with the defaults.
