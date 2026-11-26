# GRPO's per-group std turns a 0.001 KL penalty in the reward into a unit-scale pull toward the initial model

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K in verl, the launch script regularises the policy with a KL penalty in the reward at coefficient 0.001, which changes the 0/1 reward by at most 0.0030 on average. GRPO divides each prompt group's rewards by the group's standard deviation, so in every group whose eight answers are all right or all wrong the penalty is the only variation and becomes a unit-scale advantage that pulls the policy back toward the initial model. The training score stays near 0.46 and held-out accuracy goes from 0.4299 to 0.4913; the same recipe without the division reaches a training score of 0.66 and 0.5534 held-out.

| | |
| --- | --- |
| **Author** | pkulium |
| **Affiliation** | Iowa State University |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `kl-in-reward` `group-normalization` `advantage-scale` `silent-stall` |
| **Framework** | verl 0.8.0 (GRPO, naive reward manager), FSDP + colocated vLLM, `run_grpo_gsm8k.sh` |
| **Run time** | ~45 minutes |
| **Resources** | 1 × A100 (80 GB) |

## Difficulty

No code is changed. The workspace is a plain verl 0.8.0 checkout plus a GSM8K preparation script, a
reward that returns only the 0/1 answer accuracy, and the launch script. The launch script contains:

```bash
  algorithm.adv_estimator=grpo \
  ...
  algorithm.use_kl_in_reward=True \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
```

Each option is a documented verl switch with verl's default value for the coefficient. The penalty is
tiny in the logs: `critic/score/mean − critic/rewards/mean` is at most 0.0030 at any step.

Two unmodified pieces of verl then meet:

1. `verl/trainer/ppo/ray_trainer.py::apply_kl_penalty` subtracts `0.001 × (log π − log π_ref)` from the
   reward of every response token.
2. `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` sums the token rewards of each sample
   and, because `algorithm.norm_adv_by_std_in_grpo` is left at its default `True`, standardises them
   inside each prompt group:

```python
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

In a group where some answers are right and some are wrong, the std is about 0.35-0.53 and the penalty
is noise. In a group where all eight answers are right, or all eight are wrong, accuracy is constant.
The only variation left is 0.001 × the summed log-ratio of each sample to the initial model, and the
division scales it to unit variance. Those groups get advantages as large as any other group's, and the
advantage is highest for the sample the policy has moved toward the least. In expectation that is a
unit-scale push toward the initial model (the k1 estimate of −∇KL, rescaled), and because the log-ratio
is summed over tokens it also favours shorter answers. The coefficient drops out: 0.0001 would give the
same advantages there. (At step 1 the policy equals the reference, the log-ratio is 0 and those groups
get zero advantage; the effect starts at step 2.)

In the seven rollout dumps shipped with the run, 13-27 of the 64 groups per step are of that kind
(20-42% of the samples; over all 60 steps of the original run, whose other dumps are not
shipped, the range is 11-29):

| step | groups whose 8 answers are all right or all wrong | training accuracy |
| ---: | ---: | ---: |
| 1 | 18 of 64 | 0.326 |
| 10 | 13 of 64 | 0.500 |
| 20 | 23 of 64 | 0.463 |
| 30 | 20 of 64 | 0.461 |
| 40 | 21 of 64 | 0.523 |
| 50 | 19 of 64 | 0.547 |
| 60 | 27 of 64 | 0.459 |

Nothing fails loudly. The run looks like a slow but stable one:
- no error, no NaN; `actor/grad_norm` stays at 0.64-0.90 and `actor/pg_clipfrac` at 0.0007-0.0014;
- `critic/score/mean` rises from 0.326 at step 1 to a mean of 0.464 over steps 11-60 and stops there
  (mean of steps 51-60: 0.480; the control: 0.661);
- the policy stays close to the initial model: `actor/reward_kl_penalty` (mean per-token log-ratio) is
  0.0105 at step 60, against 0.1002 in the control;
- `actor/entropy` is still 0.344 at step 60 (0.093 in the control), and `response_length/mean` goes
  322 → 299 tokens (325 → 284);
- held-out accuracy starts at 0.4299, peaks at 0.4913 (step 30) and ends at 0.4913.

Why the verdict is "unhealthy" from the workspace alone, although accuracy rises a little:
- mainly, the configured regulariser is not the one being applied: the KL term is at most 0.0030 of the
  reward, yet in 13-27 of the 64 groups per step in the kept dumps (20-42% of the samples) it is
  the entire advantage, at the same scale as the accuracy signal in the other groups. The coefficient has
  no effect there;
- consistent with that, the policy has barely moved: 3.1 nats of summed log-ratio per response to the
  initial model at step 60, with entropy still 0.344;
- and the run has flattened at a low level: over steps 31-60 `critic/score/mean` gains 0.0052 per 10
  steps and never exceeds 0.574, and held-out accuracy stays within 0.4852-0.4913. On its own a
  plateau would not prove a defect.

| held-out accuracy | step 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4299 | 0.4556 | 0.4647 | 0.4913 | 0.4852 | 0.4875 | 0.4913 |
| defective, seed 43 | 0.4284 | 0.4443 | 0.4723 | 0.4814 | 0.4958 | 0.4890 | 0.4943 |
| control, seed 42 | 0.4299 | 0.4814 | 0.5080 | 0.5307 | 0.5459 | 0.5337 | 0.5534 |
| control, seed 43 | 0.4268 | 0.4602 | 0.5034 | 0.5118 | 0.5133 | 0.5201 | 0.5398 |
| plain GRPO, no KL term, seed 42 | 0.4412 | 0.4716 | 0.4882 | 0.5193 | 0.5262 | 0.5254 | 0.5186 |

| seed 42 | defective (shipped) | control |
| --- | --- | --- |
| `critic/score/mean`, step 1 → mean of steps 51-60 | 0.326 → 0.480 | 0.312 → 0.661 |
| `actor/reward_kl_penalty` at steps 10 / 30 / 60 | 0.0028 / 0.0067 / 0.0105 | 0.0206 / 0.0702 / 0.1002 |
| mean KL per response at step 60 (per-token × length, nats) | 3.14 | 28.42 |
| largest `score − rewards` gap, steps 1-60 | 0.0030 | 0.0286 |
| `actor/entropy` at step 60 | 0.344 | 0.093 |
| `response_length/mean`, step 1 → 60 (tokens) | 322 → 299 | 325 → 284 |
| `actor/grad_norm` range | 0.64-0.90 | 0.26-0.40 |

The control is the same tree with one more line in the launch script,
`algorithm.norm_adv_by_std_in_grpo=False`. The KL penalty in the reward and its 0.001 coefficient are
unchanged. Its penalty term grows larger than the defective run's, because its policy moves further, and
it still learns. This control also rescales the groups in which accuracy varies (their advantages are no
longer divided by 0.35-0.53; `critic/advantages/max` is 0.88 against 2.47, and
`actor/grad_norm` is about 2.5× smaller), so it is not a pure isolation; the rescale works against the
hypothesis, since the control takes smaller steps and still learns more.

A second reference arm isolates the other side: plain GRPO with the per-group std left on and `algorithm.use_kl_in_reward=False` (no KL term anywhere; seed 42). It reaches 0.5186 held-out at step 60 (0.5262 / 0.5254 / 0.5186 at steps 40/50/60) and a training score of 0.647 over steps 51-60 (its step-0 evaluation reads 0.4412, against 0.4299 for the identical start of the other seed-42 runs: greedy vLLM decoding is not bit-deterministic, and this arm loads no reference model). So neither the std division alone nor the KL term at its nominal size stalls the run the way the combination does, although on held-out the plain arm is only about 3 points above the defective runs (one seed). From step 30 on, every evaluation of the 3 runs without the defect is at least 0.5118 and every evaluation of the 2 defective runs is at most 0.4958.

Is this a defect or a design choice? verl accepts the combination and its default is
`use_kl_in_reward=False`; the GRPO paper puts the KL term in the loss rather than the reward so that it
does not enter the advantage, while REINFORCE++-style recipes do put it in the reward and verl
supports that. The point of the task is that the configured strength is not the applied strength: the
coefficient (0.001, a 0.0030 change to the reward) has no effect in the groups where the term matters,
and nothing in the logs reports this.

Wrong diagnoses:
- "the KL coefficient is too high": it is verl's default 0.001, it changes the reward by at most 0.0030, the coefficient cancels in the groups that matter, and the control keeps it.
- "learning rate too low" or "too few steps": the control uses the same 2e-6 and the same 60 steps.
- "PPO clipping holds the policy": `actor/pg_clipfrac` stays at 0.0007-0.0014.
- "the reward or the answer extraction is wrong": the reward file returns verl's own GSM8K score, and `acc` matches a numeric re-score of the held-out dumps within 0.021 from step 10 on.
- "healthy, just slow".
- naming `use_kl_in_reward` without the per-group std: that points at the right switch but does not say why a term of size 0.0030 can hold back a 0/1 reward.

The declared image ships no verl package (`pip show verl` finds nothing), so the vendored copy cannot be
diffed against an installed one.

Horizon: 60 steps of 64 prompts × 8 samples, with two mini-batch updates per step. Greedy evaluation
runs on all 1,319 test questions at step 0 and every 10 steps.

The task is filed under `configuration` because the defect is a pair of launch-script settings (one set,
one left at its default) that verl accepts and that do not do what their values suggest together.

Issue-only: the step-60 gap is 0.062 for seed 42 and 0.045 for seed 43 (two seeds per arm, one run each), and the defective run still improves by
0.061, so no rerun threshold was calibrated.

## Reference solution

- **issue**: a passing diagnosis must name this chain: the KL penalty is applied in the reward
  (`algorithm.use_kl_in_reward=True`) → GRPO sums it into each sample's reward and standardises inside
  each prompt group (`norm_adv_by_std_in_grpo`, default `True`) → in groups whose answers are all right or
  all wrong the penalty is the only variation, so it is scaled to a unit-scale advantage that favours
  staying at the initial model → the policy is held near the initial model and held-out accuracy stalls.
  Blaming the KL penalty or its coefficient without the group standardisation earns partial credit. The
  reference fix is one launch-script line, `algorithm.norm_adv_by_std_in_grpo=False`
  (`evaluations/issue/reference/fix.diff`). Taking the KL term out of the reward is the other repair
  (the recorded plain-GRPO arm, `algorithm.use_kl_in_reward=False`, learns as well; moving the term
  into the loss with `actor.use_kl_loss=True` was not run).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Passing needs a verdict score of 1.0 and both other scores above 0.8.
- The recorded comparison is in `evaluations/issue/reference/RESULTS.md`: two seeds per arm, one run each, so the margins
  are not a variance estimate.

All five runs used the declared image digest (`verlai/verl@sha256:4c43bbf1…`) on one A100-SXM4-80GB
that other jobs were using at the same time, with the workspace mounted at `/workspace`. They ran under
Apptainer 1.4.0 (`apptainer exec --nv`, SIF converted from the same digest), not Docker.

Startup check (2026-11-25): `python3 -m autorlbench rerun --backend local`, run inside the declared
image under Apptainer on one A100, on a copy of this task whose only difference is
`requires_isolated_stages = false` (the local backend refuses isolated tasks, and this machine has no
Docker). `prepare.sh` exited 0 (both pinned assets fetched and hash-checked), the `run_spec.json`
command started, the step-0 validation read 0.4321, and training reached step 5 of 60; I then stopped it
by hand, so no `report.json` was written. Not verified: a Docker run, and `rerun` on the unmodified
task path.
