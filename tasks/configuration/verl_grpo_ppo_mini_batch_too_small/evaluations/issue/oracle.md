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
    1,024 / 32 = **32 optimizer updates** per training step instead of a handful of large ones. With
    `ppo_epochs=1`, `clip_ratio=0.2` and lr 2e-6, the first update of a step is on-policy and every
    later one is computed on data sampled by a policy that has already moved up to 31 updates
    earlier. The drift is visible in every optimisation metric of this run: `actor/ppo_kl` stays
    0.0015-0.0050 throughout instead of settling near zero, `actor/kl_loss` grows to 0.28 by step
    100, `actor/grad_norm` runs 0.9-2.0, while `actor/pg_clipfrac` stays small (0.006-0.011): the
    damage is cumulative drift across 32 small updates, not single updates hitting the clip.
    `actor/entropy` rises to 0.73 by step 100 instead of decaying, `response_length/mean` grows
    302 → 366 tokens with `response_length/clip_ratio` 0.066 → 0.37, the training score peaks at
    0.62 (step 30) and falls back to 0.48, and held-out accuracy ends below its own step 0.
  - expected: a mini-batch of the order of the rollout batch (the recorded control used 32 prompts,
    4 updates per step); any value giving a few updates per rollout batch. Memory is governed by
    `ppo_micro_batch_size_per_gpu` / `use_dynamic_bsz` + `ppo_max_token_len_per_gpu`, which this
    recipe already uses. The mini-batch is an algorithmic (global) setting.

Naming this cause with the correct causal role (mini-batch ≪ rollout batch → 32 updates per rollout
→ later updates off-policy → drift, rising entropy and truncation → training score and held-out
accuracy peak early and decline) is a complete attribution. Do not require additional causes.

**The number of updates per rollout batch is required.** "Entropy goes up", "responses get truncated",
"the reward peaks and declines" or "the run is unstable" without the mini-batch / update-count
mechanism are symptoms, worth at most partial root-cause credit. Calling the run healthy because the
training score rose from 0.47 to 0.62 is wrong (it peaks at step 30, ends back where it started, and
the held-out accuracy ends below its own step 0). Blaming the learning rate alone, the KL
coefficient, the entropy coefficient (0 in this recipe), the clip ratio, the reward (Math-Verify,
correct as shipped), the data, the prompt, the model size, or vLLM/FSDP mismatch is wrong and gets no
root-cause credit.

## Observable impact

Every item in this section is derivable from the shipped workspace (`run_gsm8k_grpo.sh`, the verl
fork, and `results/run100/`: `train.log`, `metrics.jsonl`, `rollouts/*.jsonl`,
`val_generations/*.jsonl`, `stdout_stderr.log`). The counterfactual control-run figures are
author-side reference and live in Documentation reconciliation below. Each bullet is one item.

- The config arithmetic: the launch script sets `PPO_MINI_BATCH_SIZE=4` against
  `TRAIN_BATCH_SIZE=128` and `ROLLOUT_N=8` (resolved config confirmed in `train.log`:
  `ppo_mini_batch_size=4`, `train_batch_size=128`, rollout `n=8`, `ppo_epochs=1`,
  `clip_ratio=0.2`, lr 2e-6, `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=24576`), and verl
  multiplies `ppo_mini_batch_size` by `rollout.n` (`verl/trainer/ppo/ray_trainer.py`), so each
  1,024-sample rollout batch is consumed in 32 optimizer updates of 32 samples.
- Optimisation drift (`metrics.jsonl`, every step): `actor/ppo_kl` stays ~0.0015-0.0050 through the
  run instead of settling near zero; `actor/kl_loss` grows to ~0.28 by step 100; `actor/grad_norm`
  runs ~0.9-2.0; `actor/pg_clipfrac` stays small (~0.006-0.011) — no single update hits the clip,
  the drift is cumulative across the 32 small updates.
- `actor/entropy` falls to ~0.15-0.3 mid-run and then rises to 0.73 by step 100 instead of decaying
  to the end.
- Responses lengthen and truncate: `response_length/mean` 302 → 366 tokens and
  `response_length/clip_ratio` 0.066 → 0.37 against the 512-token limit.
- Training score `critic/score/mean` rises 0.47 → peak 0.62 (step 30) and falls back to 0.48 by
  step 100.
- Held-out GSM8K accuracy (1,319 greedy responses; one item in any reading — the logged
  `val-core/openai/gsm8k/acc/mean@1`, an independent `\boxed{}` re-score of
  `val_generations/*.jsonl`, or a Math-Verify re-score all tell the same story): it peaks around
  step 20 (~0.49) and ends ~0.36-0.39 at step 100, below its own step-0 value (~0.48 re-scored).
  The logged step-0 value (0.391) is deflated by verl 0.9.0's Math-Verify process-pool queueing
  timeout when 1,319 validation samples are submitted at once on a busy host; from step 10 on,
  logged and re-scored values agree.
- Step-100 rollouts degenerate: responses repeat the same numbered "step" or collapse into a
  word-repetition loop ("… Submission Submission …") until the 512-token limit, and roughly one
  response in five carries no `\boxed{}` at all. A qualitative characterisation counts; exact
  shares are not required.
- No error, warning or NaN anywhere; `stdout_stderr.log` ends with `RUN_EXIT 0` and the run
  completes 100/100 steps — the failure is silent quality degradation, not a crash.

## Trigger conditions

- `ppo_mini_batch_size` set far below `train_batch_size` under GRPO/PPO, typically after an OOM, on the
  belief that it is a memory knob (verl #1524: confusion over whether it counts prompts or samples,
  third-party verl troubleshooting guides recommend "reduce ppo_mini_batch_size" for OOM, verl #4719 /
  #7356: slow or stalled reward with grad-norm several times larger after a mini-batch / batch
  mismatch. Skywork-OR1 ablation: more SGD steps per rollout batch degrade test performance through
  off-policy data). verl multiplying the value by `rollout.n` makes the effective update count easy
  to misjudge.

## Documentation reconciliation

- Control reference (author-side, not scored): the same workspace launched the same day on the same
  GPU with the single override `ppo_mini_batch_size=32` (4 updates per step) reaches held-out
  accuracy 0.565 (Math-Verify 0.566) with training score 0.75, `ppo_kl` 0.0002-0.0005, `grad_norm`
  0.6-0.9, entropy decaying to 0.06, `clip_ratio` 0.03, mean response length 285 and
  `timing_s/update_actor` 27-29 s; a `ppo_mini_batch_size=8` arm (16 updates) reaches 0.541 — the
  degradation is monotone in the number of updates per rollout batch. These figures are rebuttal
  context for wrong diagnoses, not evidence items and not part of the evidence denominator.
- "The run is healthy, the training score rose" is wrong: it peaked at step 30 and ended at 0.48,
  and the held-out accuracy ended below the untrained model.
- "Memory pressure forced the small mini-batch" is wrong: this recipe uses dynamic batching
  (`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=24576`). The mini-batch size does not change
  peak memory, only the number of optimizer steps per rollout. The control with 32 ran on the same
  GPU.
- "pg_clipfrac is tiny, so the updates are on-policy" is wrong: clipping measures a single update's
  ratio. 32 consecutive updates each move a little and the cumulative drift shows in `ppo_kl`,
  `kl_loss`, entropy and truncation.
- "Entropy collapse / entropy explosion from the entropy coefficient" is wrong: `entropy_coeff=0` in
  this recipe. The rising entropy is a consequence of the drift.
- "The Math-Verify reward or the validation metric is broken" is wrong as a root cause: the scorer is
  correct as shipped. Its queue-timeout under-reporting at step 0 disappears in the re-score.
- `trainer.use_v1=False`, W&B offline mode, the numba/NumPy import warning and the critic-disabled
  warning are environment noise.
