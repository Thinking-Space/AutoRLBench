# Rollout weights only sync every 64 steps for throughput, the sampler goes stale, and GRPO collapses

Training Qwen2.5-1.5B-Instruct with GRPO (verl, async agent-loop vLLM rollout) on Countdown-Tasks-3to4 with the weight sync to the resident rollout replicas deferred to every 64th step (`trainer.rollout_sync_interval=64`, a throughput experiment): every rollout between syncs is sampled from a policy up to 63 steps behind the trainer, the trainer-recomputed `old_log_probs` keep the ratio/clip metrics blind to the mismatch, the training reward never leaves its initial band while the every-step control reaches 0.27 held-out accuracy — and the first mid-run sync at step 64 pushes the drifted, entropy-collapsed trainer policy into the sampler, after which the reward is exactly 0.0 with responses pinned to the 512-token cap.

| | |
| --- | --- |
| **Author** | Xuyang Yuan |
| **Affiliation** | Shanghai Jiao Tong University |
| **Type** | Configuration |
| **Domain** | Math RL |
| **Tags** | `weight-sync` `rollout-staleness` `off-policy-drift` `async-rollout` |
| **Framework** | verl (GRPO, commit `4b9c14f2` staged with the `rollout_sync_interval` knob), FSDP2 + async colocated vLLM 0.11.0, `run_grpo_stale_sync.sh` (70 steps, `test_freq=15`) |
| **Run time** | ~25 minutes |
| **Resources** | 1 × H100 (80 GB) |

## Difficulty

At a glance the run looks healthy-but-slow: `actor/pg_clipfrac` stays at or below `0.03` in every step, rollout outputs stay coherent, and the one weight sync that runs (plus the initial one) completes cleanly (`update_weights done` in `train.log`). Nothing errors. The training reward simply never takes off, which reads like "LR too low" or "hard task" — except the identical run with `SYNC_INTERVAL=1` reaches 0.27 held-out accuracy. The clip metrics are blind to the failure by construction: in this verl the trainer recomputes `old_log_probs` from the **current** actor weights after generation, so the GRPO ratio only measures within-step policy movement and never sees the sampler-trainer mismatch.

**The graded knob is staged for this task, not upstream verl.** Upstream at commit `4b9c14f2` pushes updated actor weights to the rollout after every optimization step (`verl/trainer/ppo/ray_trainer.py` calls `checkpoint_manager.update_weights(self.global_steps)` unconditionally); the workspace carries a throughput-experiment patch adding `trainer.rollout_sync_interval` (default `1`) that skips the sync on non-multiples of the interval, and `run_grpo_stale_sync.sh` sets it to `64`. The workspace additionally carries four environment-compatibility changes against that commit (optimizer offload around `optimizer.step()` in `transformer_impl.py`, forced shared-memory weight sync in `vllm_rollout.py`, a benchmark-logger close guard in `tracking.py`, and a benign result-variable refactor in `torch_functional.py`) plus one dropped subdirectory; none of them affect the graded defect.

The mechanism is invisible to every standard metric. With the sampler `q` (weights up to 63 steps old) and scores `s_p` from the current policy `p`, the expected update `E_q[R·s_p]` carries a drift term `E_q[R|prefix]·s̄(prefix)` that is identically zero on-policy but non-zero under staleness — a distillation gradient toward the policy's own stale sampler that compounds between syncs. The run enables `actor_rollout_ref.rollout.calculate_log_probs=True`, so the engine-side logprobs are recorded and `training/rollout_probs_diff_mean` (mean per-token |log q − log p|) makes the mismatch measurable: it climbs monotonically from the ~0.005 engine-vs-trainer numerical floor to `0.5312` at step 64, while the control stays at the floor for the whole run.

| run | step 1 | step 15 | step 30 | step 45 | step 60 | step 65 | step 70 | val acc @70 | probe peak (step 64) | `pg_clipfrac` (step 70) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.0312 | 0.0293 | 0.0293 | 0.0234 | 0.0293 | 0.0000 | 0.0000 | 0.0000 | 0.5312 | 0.000 |
| defective, seed 43 | 0.0273 | 0.0273 | 0.0449 | 0.0234 | 0.0117 | 0.0000 | 0.0000 | 0.0000 | 0.3870 | 0.000 |
| control (k=1), seed 42 | 0.0430 | 0.1777 | 0.2285 | 0.2344 | 0.3281 | 0.2969 | 0.2637 | 0.2705 | 0.0011 (floor) | 0.003 |
| control (k=1), seed 43 | 0.0293 | 0.1348 | 0.1641 | 0.2402 | 0.3164 | 0.2793 | 0.3066 | 0.2461 | 0.0011 (floor) | 0.001 |
| severity ladder k=8, seed 42 | 0.0293 | 0.1523 | 0.3789 | 0.3770 | 0.4824 | 0.0312 | 0.0371 | (collapses) | sawtooth 0.003–0.04 | 0.010 |

The `pg_clipfrac` column is the trap: the defective run's clipping is *more* benign than the control's (0.000 vs 0.003 at step 70), because the collapsed policy barely moves within a step — the mismatch lives entirely outside what the ratio can express. The severity ladder is monotone: k=1 learns to 0.27 held-out accuracy, k=8 reaches 0.48 training reward by step 60 and then collapses to 0.04 in the late phase (even bounded staleness converts the entropy-squeezed endgame into collapse), and k=64 never learns at all — flat at its initial 0.03 band for 64 steps with entropy decaying 0.69 → 0.07 and the probe climbing 100×, until the step-64 sync replaces the sampler with the drifted policy and the reward goes to exactly 0.0 (responses pinned to the 512-token cap, entropy 0.025). Per-step metrics for all arms are in `evaluations/issue/reference/`.

Wrong or incomplete diagnoses:

- "off-policy data — tighten the clip / lower the LR / fewer epochs": the clip metrics are already benign; PPO-style ratios regularize the policy's own movement and are blind to the sampler-trainer mismatch;
- "the rollout never receives new weights / the sync is broken": the initial sync and the step-64 sync complete (`update_weights done` in `train.log`) — the weights flow correctly, just late;
- "an async framework bug", "advantage normalization", "data/reward problems": every component works as configured, and the controls learn with the same data, reward, and LR;
- "entropy collapse / exploration": the entropy decay is a *result* of the drift (distillation toward the stale sampler), not the root cause, and the control reaches similar late-phase entropy while holding reward.

## Reference solution

- **issue**: name the deferred weight sync (`trainer.rollout_sync_interval=64`) together with the mechanism in this stack: the resident rollout replicas keep sampling from a stale policy, the trainer-recomputed `old_log_probs` hide the mismatch from the ratio/clip metrics, and the drift term in `E_q[R·s_p]` distills the policy toward its own stale sampler — no learning, entropy decay, and a total collapse once the first mid-run sync pushes the drifted policy into the sampler. Naming the knob without the staleness/drift mechanism gets partial `root_cause_score`. Attributions to broken sync transport, framework bugs, off-policy data (fixable by clipping), LR, or data/reward issues do not pass.

## Verification

- **issue**: an LLM judge scores `agent_answer.md` against the hidden oracle on verdict, root cause, and evidence (`0.2 / 0.6 / 0.2`). Passing requires `verdict_score == 1.0` and both other scores above `0.8`.
