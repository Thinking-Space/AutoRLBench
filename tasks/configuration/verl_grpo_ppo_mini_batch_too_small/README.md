# GRPO mini-batch far below the rollout batch: 32 updates per step

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K where the launch script sets `ppo_mini_batch_size=4` "to fit memory" against a rollout batch of 128 prompts × 8 samples. verl takes 32 optimizer updates per rollout batch instead of 4, most of them off-policy: entropy rises instead of decaying, responses lengthen and truncate, and the training score and held-out accuracy peak around step 20 to 30 and then fall below the untrained model while nothing crashes or warns.

| | |
| --- | --- |
| **Author** | Ziqi Zhou |
| **Affiliation** | Duke University |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `ppo-mini-batch` `off-policy-drift` `peak-then-decline` |
| **Framework** | verl 0.9.0 (GRPO), FSDP + vLLM 0.28 colocated |
| **Run time** | ~2.7 hours per run |
| **Resources** | 1 GPU |

## Difficulty

The recipe is verl's GSM8K GRPO quickstart scaled to one GPU with a MATH-style `\boxed{}` prompt and
verl's built-in Math-Verify scorer as the reward (correct in both arms). One line of the launch script
is the defect: `PPO_MINI_BATCH_SIZE=4  # smaller mini-batches to fit memory`, where the recipe uses 32.
verl counts `ppo_mini_batch_size` in prompts and multiplies it by `rollout.n` (verl #1524 is eleven
comments of users discovering this), so the 1,024 samples of each rollout batch are consumed in
32 updates of 32 samples instead of 4 updates of 256. With `ppo_epochs=1` and `clip_ratio=0.2` the
first update of each step is on-policy. The thirty-one that follow are computed on samples drawn
from a policy up to thirty-one updates old. Memory, the stated reason, is governed by
`ppo_micro_batch_size_per_gpu` / `use_dynamic_bsz` + `ppo_max_token_len_per_gpu`, which the script
already sets. The mini-batch changes the algorithm, not the peak memory.

Nothing fails. The run completes 100 steps with `RUN_EXIT 0`, the training score rises from 0.47 to
0.62 by step 30, and the logged held-out accuracy rises from 0.39 to 0.49 by step 20. The decline is
gradual: training score 0.46 at step 50 and 0.48 at step 100 (control 0.75), held-out accuracy 0.46
(50) → 0.41 (60) → 0.35 (70) → 0.36 (100), below the untrained model's 0.48. The optimisation metrics
show drift rather than instability: `actor/ppo_kl` is ten times the control's (0.003 to 0.004 vs
0.0002 to 0.0005), `actor/grad_norm` two to three times (1.0 to 2.0 vs 0.6 to 0.9), `actor/kl_loss` 0.28 vs
0.09 at step 100, while `actor/pg_clipfrac` stays small (0.006 to 0.010 vs 0.001 to 0.002) because each of
the 32 updates moves a little: the damage is cumulative, not a single update hitting the clip (so the
0.4 clip fractions of verl #7356 are not the signature here). Entropy decays to 0.18 by step 20 as in
the control and then climbs back to 0.49 (90) and 0.73 (100), where the control's is 0.06. Responses
lengthen from 300 to 366 tokens and 37% hit the 512-token limit (control 3%): step-100 rollouts repeat
numbered "steps" that recompute the same quantity until cut off, and 23% of step-100 rollouts (235/1,024)
and 43% of held-out responses (569/1,319; control 3%) have no `\boxed{}` answer. The actor update also
takes longer (35 s vs 27 s per step).

A same-day, same-node control with `ppo_mini_batch_size=32` (4 updates per step) reaches 0.565. A
dose-response run with 8 (16 updates) reaches 0.541, so the degradation is monotone in the number of
updates per rollout batch. This is the Skywork-OR1 finding (more SGD steps per rollout batch degrade
test performance through off-policy data, entropy collapsing or inflating within a few steps) and the
situation behind verl #4719 (reward stalls, grad-norm several times larger after a batch / mini-batch
mismatch). It is also what the circulated verl troubleshooting advice "reduce `ppo_mini_batch_size`
when you OOM" produces. The bug is not in verl: it is one number in the user's launch script and a
misunderstanding of what it controls.

| step | 0 | 10 | 20 | 30 | 50 | 70 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective held-out, `\boxed{}` rule (grader) | 0.482 | 0.487 | 0.494 | 0.476 | 0.456 | 0.353 | 0.361 |
| defective held-out, Math-Verify re-score | 0.488 | 0.488 | 0.494 | 0.483 | 0.466 | 0.373 | 0.387 |
| dose mini-batch 8, `\boxed{}` rule | 0.478 | 0.496 | 0.518 | 0.519 | 0.533 | 0.547 | 0.541 |
| control mini-batch 32, `\boxed{}` rule | 0.487 | 0.489 | 0.547 | 0.543 | 0.552 | 0.572 | 0.565 |
| defective `critic/score/mean` (train) | n/a | 0.542 | 0.557 | 0.621 | 0.461 | 0.495 | 0.483 |
| control `critic/score/mean` | n/a | 0.565 | 0.646 | 0.747 | 0.624 | 0.706 | 0.745 |
| defective `actor/ppo_kl` | n/a | 0.0028 | 0.0038 | 0.0030 | 0.0031 | 0.0036 | 0.0027 |
| control `actor/ppo_kl` | n/a | 0.0002 | 0.0003 | 0.0001 | 0.0004 | 0.0002 | 0.0004 |
| defective `actor/entropy` | n/a | 0.22 | 0.18 | 0.22 | 0.25 | 0.15 | 0.73 |
| control `actor/entropy` | n/a | 0.18 | 0.13 | 0.09 | 0.09 | 0.07 | 0.06 |
| defective `response_length/clip_ratio` | n/a | 0.07 | 0.05 | 0.11 | 0.21 | 0.32 | 0.37 |
| control `response_length/clip_ratio` | n/a | 0.05 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 |

A note on the logged metric. At step 0 the logged `val-core/openai/gsm8k/acc/mean@1` is 0.391
(defective) and 0.398 (control) although the same dumps re-score to 0.488 / 0.491 with the same
Math-Verify function (0.482 / 0.487 under the grader's `\boxed{}` rule): verl 0.9.0's `math_verify.py` runs a 4-worker process pool and calls
`future.result(timeout=30)`, whose 30 s include time spent queued behind the other 1,318 validation
samples, so on a busy host samples deep in the queue are silently scored 0 (128 of 1,319 at step 0
here, 9 on a quiet node two days earlier). Training rollouts are scored in smaller batches and are
unaffected (0 disagreements), and from step 10 on the logged and re-scored values agree, so the
experiment is intact. But the logged number is load-dependent, which is why the grader re-scores the
dump instead of reading it. The side finding is not this task's defect and affects both arms equally.
The two re-scored series in the table above are also not identical: the grader's `\boxed{}` rule and
the Math-Verify re-score disagree on a few tens of responses per evaluation (up to 0.026 on the
defective arm at step 100, Math-Verify the higher), and `recorded_comparison/RESULTS.md`'s "re-score"
columns are the Math-Verify series; the binding scorer is the grader's rule.

Wrong diagnoses a reviewer should expect and reject:

- **"The run is healthy: the training score rose from 0.47 to 0.62."** It then fell to 0.48 against the
  control's 0.75, and held-out accuracy ended below step 0. Peak-then-decline is the signature.
- **"Memory forced the small mini-batch. It is a necessary trade-off."** Dynamic batching already bounds
  memory. The control with 32 ran on the same 48 GB GPU with the same settings.
- **"Entropy explosion from the entropy coefficient."** `entropy_coeff=0` in both arms. The rising
  entropy is downstream of the off-policy drift.
- **"`pg_clipfrac` is tiny, so the updates are on-policy."** Clip fraction measures one update's ratio.
  the drift is cumulative over 32 updates and shows in `ppo_kl`, `kl_loss`, entropy and truncation.
- **"The reward / logged validation metric is broken."** Math-Verify is correct and identical in both
  arms. Its queue-timeout under-reporting at step 0 is noise in the log, not the cause.
- **Learning rate alone, KL coefficient, clip ratio, step budget, model size, data, prompt.** Shared with
  the control. Lowering the learning rate to tolerate 32 updates is not the fix.

## Reference solution

- **issue**: name `PPO_MINI_BATCH_SIZE=4` (`actor_rollout_ref.actor.ppo_mini_batch_size`) against the
  128 × 8 rollout batch, i.e. 32 optimizer updates per rollout batch, as the reason most updates are
  off-policy and the run drifts: rising `ppo_kl` / `grad_norm`, rising entropy, truncation, training
  score and held-out accuracy peaking early and declining. Symptoms alone earn partial credit.
  "healthy", LR/KL/entropy-coefficient/reward/data explanations fail.
- **solution**: set the mini-batch to the order of the rollout batch (`evaluations/solution/oracle/fix.diff`:
  `PPO_MINI_BATCH_SIZE=32`, 4 updates per step). Any value giving a few updates per rollout batch
  without changing the rollout batch, epochs or learning rate passes if the rerun clears the threshold
  (8 reaches 0.541 in the recorded dose run). 4 does not.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. A run passes only if the verdict score is 1.0 and the root-cause and evidence scores are both above 0.8. The weighted score is reported but does not decide the pass.
- **solution**: the agent submits `report.md` and does not train. The harness reruns the experiment
  per `run_spec.json` (`run_gsm8k_grpo.sh`, 100 steps, 4 h budget) and grades only that run. The grader
  does not read the logged metric (see the note above) and instead re-scores the run's own
  `experiment_logs/val_generations/100.jsonl` with a fixed pure-Python rule (the number in the last
  `\boxed{}` numerically equal to `gts`, ≥ 1,300 rows). The accuracy must be ≥ 0.53. Under that rule the
  defective run is 0.361 at step 100 and 0.494 at its best (step 20). The control is 0.565 at step 100
  and 0.487 at step 0, so applying the fix without the rerun fails. The dose run with mini-batch 8 is
  0.541. Calibrated on one run per arm, so the margins (0.036 / 0.035) are not a variance estimate.
  Grader replay on the recorded dumps: defective exit 1, control exit 0. A symlinked or missing dump
  exits 1. Not verified: the `verl-0.9-vllm0.28` image is not built and `prepare.sh` has not been run
  in Docker. All recorded runs used the same stack natively. The control was launched with the hydra
  override `actor_rollout_ref.actor.ppo_mini_batch_size=32` rather than an edited script. The recorded
  launch log shows the override and the script's sha256 is identical across the three arms. A
  previously recorded control of the same recipe from a different day was not used because its step-0
  logged value differed (the Math-Verify queue effect). The pre-registration required a same-day,
  same-hardware control, which is what `recorded_comparison/control` is.
