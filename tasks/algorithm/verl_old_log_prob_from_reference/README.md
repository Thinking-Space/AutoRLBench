# Old log-probs computed by the reference model pin GRPO to the base model

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K in verl, the trainer recomputes the PPO "old" log-probs with the frozen reference model instead of the actor. The PPO clip therefore becomes a ±20% per-token leash around the base model. Nothing errors and the gradient norm looks normal, but held-out accuracy moves only from 0.4382 to 0.4647 in 60 steps, against 0.4329 to 0.5451 for the same recipe.

| | |
| --- | --- |
| **Author** | pkulium |
| **Affiliation** | Iowa State University |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `ppo-ratio` `old-log-prob` `reference-policy` `silent-stall` |
| **Framework** | verl 0.8.0 (GRPO, DAPO reward manager with overlong penalty), FSDP + colocated vLLM, `run_dapo_gsm8k.sh` |
| **Run time** | ~26 minutes |
| **Resources** | 1 × A100 (80 GB) |

## Difficulty

The defect is in `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer._compute_old_log_prob`, which sits
next to the almost identical `_compute_ref_log_prob`:

```python
        output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
```

The upstream line is `self.actor_rollout_wg.compute_log_prob(batch_td)`. The reference worker
exists because the recipe uses the KL loss (`use_kl_loss=True`, `kl_loss_coef=0.001`), so the call
succeeds and returns tensors of the right shape.

The PPO loss in `losses.py` is unmodified. It now computes π_θ / π_ref. At step 1 the actor equals
the reference, so the run starts like a healthy one. From then on, a token gets no policy gradient
once it has moved more than 20% from the base model in the direction its advantage favours. Moves
back toward the base model stay unclipped.

The same call also supplies `actor/entropy`. That metric is therefore the reference model's entropy
on the rollouts, and it stays at about 0.5 whatever the actor does. This is one more signal that
looks healthy.

Nothing crashes. `actor/grad_norm` stays at 0.74-1.05 and the clip fraction stays around 1%. The
evidence is a pattern:
- training reward and held-out accuracy barely move;
- `actor/kl_loss` never exceeds 0.0024;
- `actor/ppo_kl` (old − new log-prob) is negative at every step from step 2 and tracks
  `actor/kl_loss`. That is what "old" log-probs coming from the reference model look like. With
  the correct call, `ppo_kl` stays within ±0.0003.

Held-out accuracy below is the verl reward's `acc`, which is what `val-core/openai/gsm8k/acc/mean@1`
logs. It uses flexible extraction and a string match. A numeric re-score
(`evaluations/issue/reference/RESULTS.md`) gives the same picture. At step 60 it puts the defective
runs at 0.4845 and 0.4792, and the controls at 0.5459 and 0.5171.

| held-out accuracy | step 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4382 | 0.4466 | 0.4640 | 0.4435 | 0.4564 | 0.4602 | 0.4647 |
| defective, seed 43 | 0.4284 | 0.4450 | 0.4610 | 0.4792 | 0.4534 | 0.4572 | 0.4556 |
| control, seed 42 | 0.4329 | 0.4587 | 0.4822 | 0.5201 | 0.5254 | 0.5262 | 0.5451 |
| control, seed 43 | 0.4261 | 0.4670 | 0.4761 | 0.5216 | 0.5178 | 0.5284 | 0.5171 |

| seed 42, steps 1 → 60 | defective (shipped) | control |
| --- | --- | --- |
| `critic/score/mean` | 0.224 → 0.332 (max 0.465) | 0.229 → 0.617 (max 0.758) |
| `actor/entropy` (as logged) | 0.55 → 0.50 | 0.54 → 0.10 |
| `actor/kl_loss` at step 60 | 0.0021 | 0.082 |
| `actor/ppo_kl`, steps 10-60 | −0.0009 to −0.0019 | −0.00006 to +0.00024 |
| `actor/pg_clipfrac`, steps 10-60 | 0.0065-0.0118 | 0.0005-0.0016 |
| `actor/grad_norm` | 0.74-1.05 | 0.68-1.10 |

Wrong diagnoses:
- "learning rate too low": 2e-6 is the usual value for this model, and the control uses it.
- "KL penalty too strong": the coefficient is 0.001, and the control's KL loss grows about 40× larger.
- "clip range too tight".
- the DAPO overlong penalty: it fires only above 512 response tokens, and the control has it too.
- "the policy keeps exploring, entropy is fine": the logged entropy belongs to the reference model.
- the flexible last-number reward or the data pipeline.
- "healthy, 60 steps is too short".

The declared image ships no verl package (`pip show verl` finds nothing), so the vendored copy
cannot be diffed against an installed one. The workspace is a
plain verl 0.8.0 checkout, unmodified apart from this line, plus three small files: a GSM8K
preparation script adapted from verl's example, a reward wrapper around verl's GSM8K scorer, and the
launch script.

Horizon: 60 steps of 64 prompts × 8 samples, with two mini-batch updates per step. Greedy evaluation
runs on all 1,319 test questions at step 0 and every 10 steps.

Issue-only: at step 60 the numerically re-scored arms are 3.3-6.7 points apart (defective
0.4792-0.4845, controls 0.5171-0.5459), from one run per seed. A defective run reached 0.5027 at
step 30, and one control fell by 1.2 points over its last 10 steps. That margin is too thin for a
rerun threshold.

## Reference solution

- **issue**: a passing diagnosis must name this chain: `_compute_old_log_prob` takes the old
  log-probs from the reference worker → the PPO ratio and clip are measured against the frozen base
  model → updates that move a token more than 20% from the base model stop receiving gradient → the
  policy stays near the base model and held-out accuracy plateaus at about 0.46. "Over-clipping",
  "too slow" or "lr too low" without locating the reference call earns partial credit. The reference
  fix is to call the actor (`evaluations/issue/reference/fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Passing needs a verdict score of 1.0 and both other scores above 0.8.
- The recorded comparison is in `evaluations/issue/reference/RESULTS.md`: two seeds per arm, one run
  each, so the margins are not a variance estimate.

All four runs used the declared image digest (`verlai/verl@sha256:4c43bbf1…`) on an
A100-SXM4-80GB, with the workspace mounted at `/workspace`. They ran under Apptainer 1.4.0
(`apptainer exec --nv`, SIF converted from the same digest), not Docker. `environment/prepare.sh`
was run once in the same image; it fetched both pinned assets and their sha256 checks passed.
Not verified: a Docker run, and the harness `rerun` command.
