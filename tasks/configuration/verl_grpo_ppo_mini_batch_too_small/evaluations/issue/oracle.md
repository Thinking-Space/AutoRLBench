# Canonical issue oracle

Reconciled from the task's frozen evidence and the recorded control (same workspace, same day and GPU
type, seed and budget, the control was launched with the single override
`actor_rollout_ref.actor.ppo_mini_batch_size=32`, i.e. the value `fix.diff` writes into the script).

Expected verdict: **unhealthy**.

## Root cause

- `run_gsm8k_grpo.sh`, `PPO_MINI_BATCH_SIZE=4` → `actor_rollout_ref.actor.ppo_mini_batch_size=4`
  (comment: "smaller mini-batches to fit memory"), against `data.train_batch_size=128` and
  `actor_rollout_ref.rollout.n=8`
  - kind: `config`
  - observed: verl counts `ppo_mini_batch_size` in prompts and multiplies it by `rollout.n`
    (`ray_trainer.py`, v0.9.0), so each rollout batch of 128 × 8 = 1,024 samples is split into
    1,024 / 32 = **32 optimizer updates** per training step instead of the recipe's 4 (mini-batch 32
    prompts = 256 samples). With `ppo_epochs=1`, `clip_ratio=0.2` and lr 2e-6, the first update of a step
    is on-policy and every later one is computed on data sampled by a policy that has already moved
    up to 31 updates earlier. The drift is visible in every optimisation metric relative to the
    control: `actor/ppo_kl` 0.0016 to 0.0044 vs 0.0001 to 0.0005, `actor/grad_norm` 1.0 to 2.0 vs 0.6 to 0.9,
    `actor/kl_loss` 0.28 vs 0.09 at step 100, `actor/entropy` rising 0.35 → 0.73 instead of decaying
    to 0.06, `response_length/clip_ratio` 0.07 → 0.37 (control 0.03) with mean length 366 vs 285
    tokens. `actor/pg_clipfrac` stays small (0.006 to 0.010 vs 0.001 to 0.002): the damage is cumulative
    drift across 32 small steps, not single updates hitting the clip. Training score peaks at 0.62
    (step 30) and falls to 0.48 (control 0.75). Held-out accuracy peaks at 0.494 (step 20) and ends at
    0.361 (`\boxed{}` rule, 0.387 by Math-Verify), below its own step 0.
  - expected: a mini-batch of the order of the rollout batch: `ppo_mini_batch_size=32` (4 updates per
    step, the recipe's value, 0.565 at step 100). `ppo_mini_batch_size=8` (16 updates) reaches 0.541:
    the degradation is monotone in the number of updates per step. Memory is governed by
    `ppo_micro_batch_size_per_gpu` / `use_dynamic_bsz` + `ppo_max_token_len_per_gpu`, which this
    recipe already uses. The mini-batch is an algorithmic (global) setting.

Naming this cause with the correct causal role (mini-batch ≪ rollout batch → 32 updates per rollout
→ later updates off-policy → drift, rising entropy and truncation → training score and held-out
accuracy peak early and decline) is a complete attribution. Do not require additional causes.

**The number of updates per rollout batch is required.** "Entropy goes up", "responses get truncated",
"the reward peaks and declines" or "the run is unstable" without the mini-batch / update-count
mechanism are symptoms, worth at most partial root-cause credit. Calling the run healthy because the
training score rose from 0.47 to 0.62 is wrong (it ends below the control's and the held-out
accuracy ends below step 0). Blaming the learning rate alone, the KL coefficient, the entropy
coefficient (0 in both arms), the clip ratio, the reward (Math-Verify, correct in both arms), the
data, the prompt, the model size, or vLLM/FSDP mismatch is wrong and gets no root-cause credit.

## Observable impact

- Held-out GSM8K accuracy (1,319 greedy responses, independent `\boxed{}` re-score of
  `val_generations/*.jsonl`): defective 0.482 (step 0) → 0.494 (20) → 0.456 (50) → 0.407 (60) → 0.353
  (70) → 0.361 (100). Control 0.487 → 0.547 (20) → 0.552 (50) → 0.567 (60) → 0.572 (70) → 0.565 (100).
  Math-Verify re-score (the reward function run in the main process): defective 0.488 → 0.494 → 0.387,
  control 0.491 → 0.566.
- Logged `val-core/openai/gsm8k/acc/mean@1`: defective 0.391 (step 0) → 0.494 (20) → 0.387 (100).
  control 0.398 → 0.547 → 0.566. The step-0 values are lower than the re-score (0.482 / 0.487) because
  verl 0.9.0's Math-Verify scorer charges process-pool queueing time against its 30 s timeout and
  silently scores timed-out samples 0 when 1,319 validation samples are submitted at once on a busy
  host (128 of 1,319 at step 0 in the defective run). From step 10 on logged and re-scored values
  agree. This is noise in the logged metric, not the defect. The same scorer serves both arms.
- Training (`metrics.jsonl`, every step): `critic/score/mean` defective 0.47 → 0.62 (30) → 0.46 (50)
  → 0.48 (100). Control 0.46 → 0.75 (30) → 0.75 (100). `actor/ppo_kl` 0.0016 (step 1) → 0.0044 (80),
  control 0.0002 → 0.0005. `actor/grad_norm` 1.96 (step 1) → 1.0, control 0.64 → 0.85.
  `actor/pg_clipfrac` 0.006 to 0.010 vs 0.001 to 0.002. `actor/entropy` 0.35 → 0.18 (20) → 0.49 (90) → 0.73
  (100), control 0.37 → 0.06. `actor/kl_loss` 0.28 vs 0.09. `response_length/mean` 302 → 366 (control
  285), `response_length/clip_ratio` 0.066 → 0.37 (control 0.026). `timing_s/update_actor` 31 to 36 s vs
  27 to 29 s (32 small updates cost more than 4 large ones).
- Step-100 defective rollouts repeat enumerated "steps" ("9. **Calculate the number of eggs sold in a
  day:** ... 9. **Calculate** ...") until the 512-token limit. 37% have no `\boxed{}` at all.
- No error, warning or NaN anywhere. `RUN_EXIT 0`.

## Trigger conditions

- `ppo_mini_batch_size` set far below `train_batch_size` under GRPO/PPO, typically after an OOM, on the
  belief that it is a memory knob (verl #1524: confusion over whether it counts prompts or samples,
  third-party verl troubleshooting guides recommend "reduce ppo_mini_batch_size" for OOM, verl #4719 /
  #7356: slow or stalled reward with grad-norm several times larger after a mini-batch / batch
  mismatch. Skywork-OR1 ablation: more SGD steps per rollout batch degrade test performance through
  off-policy data). verl multiplying the value by `rollout.n` makes the effective update count easy to
  misjudge.

## Documentation reconciliation

- "The run is healthy, the training score rose" is wrong: it peaked at step 30 and ended at 0.48 against
  the control's 0.75, and held-out accuracy ended below the untrained model.
- "Memory pressure forced the small mini-batch" is wrong: this recipe uses dynamic batching
  (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=24576`). The mini-batch size does not change
  peak memory, only the number of optimizer steps per rollout. The control with 32 ran on the same GPU.
- "pg_clipfrac is tiny, so the updates are on-policy" is wrong: clipping measures a single update's
  ratio. 32 consecutive updates each move a little and the cumulative drift shows in `ppo_kl`,
  `kl_loss`, entropy and truncation.
- "Entropy collapse / entropy explosion from the entropy coefficient" is wrong: `entropy_coeff=0` in
  both arms. The rising entropy is a consequence of the drift.
- "The Math-Verify reward or the validation metric is broken" is wrong as a root cause: the scorer is
  correct and identical in both arms. Its queue-timeout under-reporting at step 0 affects both logs and
  disappears in the re-score.
- `trainer.use_v1=False`, W&B offline mode, the numba/NumPy import warning and the critic-disabled
  warning are environment noise.
