# AdamW epsilon larger than the gradients it normalizes

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K. The launcher overrides the actor's AdamW `eps` to 1e-3 "because bf16 second-moment estimates underflow", but the per-parameter gradients of this 0.5B model are of order 1e-5, so eps dominates Adam's denominator and every update is 10 to 100 times smaller than the learning rate implies. Nothing looks broken: losses, gradients and entropy are smooth. The run simply learns about 30 steps late, and held-out accuracy ends at 0.486 against 0.566 for the same run with the default eps.

| | |
| --- | --- |
| **Author** | Dengzhe Hou |
| **Domain** | configuration |
| **Tags** | `verl` `grpo` `gsm8k` `optimizer` `adamw` `epsilon` `slow-learning` |
| **Model** | Qwen2.5-0.5B-Instruct |
| **Dataset** | GSM8K (openai/gsm8k main): 7,473 train / 1,319 test |
| **Framework** | verl 0.9.0.dev @ `6f2aa24`, FSDP + vLLM colocated, `run_gsm8k_grpo.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 1 × GPU (recorded on 1 × RTX PRO 6000 96 GB) |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control run, 100 steps, eval every 10 incl. step 0, seed 42. About 1 h each |

## Difficulty

The defect is two lines in `run_gsm8k_grpo.sh`:

```bash
ADAM_EPS=0.001               # AdamW eps raised from 1e-8: bf16 second-moment estimates underflow and give spiky updates
...
    actor_rollout_ref.actor.optim.override_optimizer_config={eps:$ADAM_EPS}
```

`verl/workers/config/optimizer.py::build_optimizer` passes `override_optimizer_config` straight into
`torch.optim.AdamW`. The stated reason does not hold: the FSDP actor keeps fp32 master weights and
fp32 optimizer state (`model_dtype: fp32`; bf16 is used only for compute), so nothing underflows.
What the setting does is change the optimizer. With `actor/grad_norm` 0.2 to 1.0 over about
4.9e8 parameters, the per-parameter RMS gradient is about 1e-5 to 5e-5, so `sqrt(v_hat)` is
20 to 100 times smaller than eps. AdamW's step `lr * m_hat / (sqrt(v_hat) + eps)` becomes
`(lr / eps) * m_hat = 2e-3 * m_hat`, plain momentum SGD with steps far below the configured
learning rate, and the step now shrinks with the gradient, so it is smallest when the reward is
sparsest.

The symptom is only speed, and every per-step signal looks clean. `actor/pg_clipfrac` averages 8e-5 over
steps 1 to 10 (control 2e-3) and never exceeds 1.6e-3, `actor/ppo_kl` stays within ±2.1e-4 around
zero (control 5e-5 to 1.2e-3, always positive), and the reference KL is 30 times smaller at step 10
(0.0012, control 0.039). The policy learns the `#### N` format 30 steps late: the training score is 0.026 at
step 10 (control 0.49) and first reaches 0.3 at step 35 (control step 5). Held-out accuracy follows.
By step 100 the run has caught up on format (98% of held-out answers use `####`) but not on
accuracy.

| step | 0 | 10 | 20 | 30 | 50 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out strict accuracy | 0.001 | 0.002 | 0.102 | 0.295 | 0.465 | 0.486 |
| control held-out strict accuracy | 0.001 | 0.498 | 0.542 | 0.544 | 0.569 | 0.566 |
| defective training score | n/a | 0.026 | 0.136 | 0.210 | 0.444 | 0.574 |
| control training score | n/a | 0.493 | 0.639 | 0.667 | 0.624 | 0.686 |

The run looks like a healthy run that is a bit slow, and the obvious stories are wrong. Wrong
diagnoses to reject: the run is healthy because held-out accuracy rises from 0 to 0.49 and no metric
is unstable (the control reaches 0.50 by step 10 and ends at 0.566); the learning rate is too low
(2e-6 in both runs); the reward is too sparse at the start (the control starts from the same
untrained model and reward, with a step-1 training score of 0.016 against 0.012 here, and is at 0.49
by step 10); the
KL coefficient or clip ratio holds the policy back (shared with the control; clipping almost never
fires); the 0.5B model is at its capacity (the same model reaches 0.566); the reward function, the
data or the sampler (unchanged, the same as the control).

## Reference solution

- **issue**: name the `eps=1e-3` override as the root cause, with the mechanism: eps is far above
  `sqrt(v_hat)` at this gradient scale, so AdamW degenerates into small-step momentum SGD and the run
  learns tens of steps late and ends below the control. "Learning is slow" or "the learning rate is
  too low" without the epsilon earns partial credit. Healthy, sparse-reward, KL and capacity
  explanations fail.
- **solution**: remove the override so AdamW uses its default eps 1e-8 (`fix.diff`). Any eps well
  below the per-parameter gradient scale is equivalent; raising the learning rate to compensate is
  a different optimizer and was not recorded.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. A run passes only if the verdict score
  is 1.0 and the root-cause and evidence scores are both above 0.8.
- **solution**: the agent submits `report.md` and does not train. The harness reruns the experiment
  per `run_spec.json` (`run_gsm8k_grpo.sh`, 100 steps, 3 h budget) and grades only that run. The
  grader re-scores `experiment_logs/val_generations/100.jsonl` with a fixed strict rule (last
  `#### N` equals the gold answer, at least 1,300 rows) and requires ≥ 0.53. The grader's rule also
  accepts forms such as `#### 18.00`, so it scores the defective step 100 at 0.493 (650 of 1,319;
  verl's logged strict metric is 0.486) and the control at 0.566 (746). The control's step 0 is
  0.001, so fixing the launcher without the rerun fails. Margins 0.037 to the defective final and
  0.036 to the control, calibrated
  on one run per side, so they are margins, not a variance estimate. The repair is in the runner, so
  the runner is not byte-compared (`runner_pinned: false`, as in
  `configuration/verl_grpo_ppo_mini_batch_too_small`); the GSM8K parquets and raw files are pinned
  by sha256 and `metrics.jsonl` must show training steps 0 to 100. Grader replay on the recorded
  evidence: defective exit 1, control exit 0. The recorded runs used the `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image (same digest) through
  Apptainer.

Not verified: a Docker build and run of the task (the runs and checks used Apptainer on the same image
digest), and a separate rerun of `fix.diff` (the control run is the workspace with `fix.diff` applied,
recorded once).
