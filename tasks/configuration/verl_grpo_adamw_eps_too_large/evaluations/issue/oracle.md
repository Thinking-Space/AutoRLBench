# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed and budget; only the `ADAM_EPS` override in
`run_gsm8k_grpo.sh` differs: eps 1e-3 here, AdamW's default 1e-8 in the control).

Expected verdict: **unhealthy**.

## Root cause

- `run_gsm8k_grpo.sh` (`ADAM_EPS=0.001` → `actor_rollout_ref.actor.optim.override_optimizer_config={eps:0.001}`)
  - kind: `config`
  - observed: `build_optimizer` passes the override into `torch.optim.AdamW`, so the actor trains with
    eps 1e-3. The per-parameter RMS gradient of this 0.5B model is about 1e-5 to 5e-5
    (`actor/grad_norm` 0.2 to 1.0 over about 4.9e8 parameters), so eps dominates the denominator
    `sqrt(v_hat) + eps` and the AdamW step becomes `(lr / eps) * m_hat`: momentum SGD with steps one
    to two orders of magnitude below what lr 2e-6 gives with the default eps, shrinking further when
    gradients are small. The comment's reason (bf16 second moments underflow) does not apply: the
    FSDP actor keeps fp32 master weights and fp32 optimizer state (`model_dtype: fp32`).
  - expected: AdamW's default eps (1e-8), or any eps well below the per-parameter gradient scale.

Naming this override with its causal role (eps far above `sqrt(v_hat)` → Adam's normalization
removed and updates throttled → the format and accuracy are learned about 30 steps late → held-out
accuracy at step 100 well below what the same recipe reaches) is a complete attribution. Do not
require additional causes.

**The setting and the mechanism are required.** "Learning is slow", "the learning rate is too low"
or "the reward is too sparse at the start" without tracing it to the eps override earns at most
partial root-cause credit. Calling the run healthy because held-out accuracy rises from 0 to about
0.49 and no metric is unstable, or blaming the learning rate, the KL coefficient, the clip ratio,
the model's capacity, the reward function, the data or the sampler, is wrong and gets no root-cause
credit.

## Observable impact

Every item here is derivable from the shipped workspace (`results/run100` plus the code and
configuration).

- The override is active: `run_gsm8k_grpo.sh` sets `ADAM_EPS=0.001` and passes
  `actor_rollout_ref.actor.optim.override_optimizer_config={eps:0.001}`, which
  `verl/workers/config/optimizer.py::build_optimizer` forwards to `torch.optim.AdamW`.
- Per-update policy change is tiny for the whole run: `actor/pg_clipfrac` averages 8e-5 over steps 1
  to 10 and never exceeds 1.6e-3; `actor/ppo_kl` stays within ±2.1e-4 around zero; `actor/kl_loss`
  is 0.0012 at step 10, 0.021 at step 30 and 0.089 at step 100, with `actor/grad_norm` 0.2 to 1.0 and
  rarely clipped.
- Learning is delayed: training score `critic/score/mean` 0.012 (step 1) → 0.026 (10) → 0.136 (20) →
  0.210 (30), first at least 0.3 at step 35, 0.574 at step 100. Held-out
  `val-core/openai/gsm8k/acc/mean@1` (equal to a strict re-score of `val_generations`): 0.001 (0) →
  0.002 (10) → 0.102 (20) → 0.295 (30) → 0.400 (40) → 0.465 (50) → 0.486 (100); `####` appears in
  30 of 1,319 held-out answers at step 0, 1,247 at step 50 and 1,292 at step 100.
- No instability: entropy 0.55 → 0.16 without spikes, response-length clip ratio 0.12 → 0.02, no
  group of identical rollouts, mixed-script garbage in at most 4 of 1,024 rollouts (as in the
  control), rollout/actor log-prob correlation 0.998 or higher.

## Trigger conditions

- An Adam epsilon of the same order as, or larger than, the square root of the second-moment
  estimate, which for a large model trained with small per-parameter gradients happens far below
  the eps values used in small-network RL (1e-5 to 1e-3).
- Nothing in verl validates `override_optimizer_config`, and no per-step metric reports the update
  size, so the only symptom is slower learning.

## Documentation reconciliation

- Control reference (author-side; same code, data, seed and budget with the default eps; not shipped
  in the workspace, rebuttal context only): held-out strict accuracy 0.001 → 0.498 (10) → 0.542 (20)
  → 0.544 (30) → 0.569 (50) → 0.566 (100); training score 0.016 (1) → 0.493 (10), first at least 0.3
  at step 5, 0.686 at step 100; `pg_clipfrac` 2e-3 on average over steps 1 to 10; `kl_loss` 0.039 at
  step 10.
- "The run is healthy" is wrong: the same recipe reaches 0.50 by step 10 and ends 0.08 above this
  run; the shipped `examples/ppo_trainer/README.md` also lists 56.7 for PPO on this model.
- "The learning rate is too low" is wrong as a root cause: lr 2e-6 is the same in the control; the
  effective step is small because of eps.
- "The reward is too sparse at the start" is wrong: the control starts from the same model and reward
  (step-1 training score 0.016 against 0.012 here) and is at 0.49 by step 10.
- The reward function, the DAPO reward manager (overlong penalty disabled), the preprocessing, the
  sampler (`top_p=1.0`, temperature 1.0), GRPO advantage, KL coefficient, clip ratio and loss
  aggregation are unchanged from the control.
- "Training did not improve" is wrong: held-out strict accuracy rises from 0.001 to 0.486 and the
  training score to 0.57. The defect makes the run learn about 30 steps late and end below the
  control (0.486 against 0.566); it does not stop learning.
- The ray/vLLM warnings, `trainer.use_v1=False`, W&B offline mode and the wandb teardown traceback
  at exit are environment noise.
