# GRPO advantage gather misaligns prompts while KL straight-through squares sequence ratios

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, `grouped_whiten` indexes standardized advantages with the forward sort order instead of inverting it, and `kl_penalty` routes `low_var_kl` into a straight-through branch that squares the sequence-summed log-ratio. Forward `actor/kl_loss` looks small and clean, but backward KL gradients overwhelm policy updates and hold step-50 accuracy at 0.4610 against 0.5588.

| | |
| --- | --- |
| **Author** | Bo Chen |
| **Affiliation** | University of Hong Kong |
| **Type** | Algorithm |
| **Domain** | Computer Science |
| **Tags** | `grpo` `advantage-estimation` `permutation-bug` `kl-penalty` `straight-through-estimator` `gradient-scaling` |
| **Framework** | verl (GRPO, commit `4b9c14f2`), FSDP + vLLM 0.11.0 colocated, `run_grpo_gsm8k.sh` (50 steps, `test_freq=25`) |
| **Run time** | ~24 minutes |
| **Resources** | 1 × A800 (80 GB) |

## Difficulty

The first bug is in `verl/utils/torch_functional.py::grouped_whiten`, called from `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`:

```python
order = torch.argsort(group_ids, stable=True)
...
whitened = torch.cat(whitened_groups, dim=0)
return torch.gather(whitened, 0, order)
```

`order` sorts rows by prompt UID. Calling `torch.gather(whitened, 0, order)` applies the forward permutation a second time instead of scattering values back to their original slots (`torch.empty_like(whitened).scatter_(0, order, whitened)` or `whitened.index_select(0, torch.argsort(order))`). With batch sequence-length balancing on, samples from the same prompt are interleaved (`order` is not its own inverse), so trajectories get another row's advantage while batch mean (~0) and std (~1) still look fine.

On its own, that misassignment makes the policy drift downhill (`0.4102` / `0.3654` by step 50). A second bug in `verl/trainer/ppo/core_algos.py::kl_penalty` hides the collapse:

```python
forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
if kl_penalty in ("kl", "k1", "abs", "mse", "k2"):
    return forward_score

seq_log_ratio = (logprob - ref_logprob).sum(dim=-1, keepdim=True)
backward_score = 0.5 * seq_log_ratio.square().expand_as(logprob)

return backward_score - backward_score.detach() + forward_score.detach()
```

Dropping `not kl_penalty.endswith("+")` sends `low_var_kl` into the straight-through branch, where `backward_score` squares the sequence sum and broadcasts it across every token. Forward `actor/kl_loss` (`0.00795`) and `actor/loss == actor/pg_loss + 0.001 * actor/kl_loss` (residual `4.17e-9`) stay unchanged, while the backward KL gradient grows by ~`L_i` per token (~`L_i^2` per sequence, around `10^4–10^5×` at `L_i ≈ 280–300`). That keeps the policy pinned near the base model: held-out accuracy stays flat (`0.4625` → `0.4579` → `0.4610`), `actor/pg_clipfrac` stays under `0.001`, and even after fixing `grouped_whiten` alone the run only reaches `0.4807` / `0.4905`.

| run | step 0 | step 25 | step 50 (re-scored) | logged `val-core` (step 50) | `actor/kl_loss` (step 50) | `actor/entropy` (step 1 → 50) | `critic/score/mean` (step 1 → 50) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.4625 | 0.4579 | 0.4610 | 0.4337 | 0.00795 | 0.485 → 0.456 | 0.4102 → 0.3594 |
| defective, seed 43 | 0.4625 | 0.4617 | 0.4640 | 0.4375 | 0.00635 | 0.478 → 0.419 | 0.3867 → 0.3691 |
| fix `kl_penalty` only, seed 42 | 0.4625 | 0.4481 | 0.4102 | 0.3578 | 0.05804 | 0.497 → 0.398 | 0.4160 → 0.3164 |
| fix `kl_penalty` only, seed 43 | 0.4632 | 0.4261 | 0.3654 | 0.3632 | 0.07011 | 0.497 → 0.220 | 0.3418 → 0.4004 |
| fix `grouped_whiten` only, seed 42 | 0.4617 | 0.4829 | 0.4807 | 0.4693 | 0.00748 | 0.526 → 0.393 | 0.3906 → 0.4180 |
| fix `grouped_whiten` only, seed 43 | 0.4647 | 0.4754 | 0.4905 | 0.4754 | 0.00756 | 0.476 → 0.355 | 0.3535 → 0.4473 |
| control, seed 42 | 0.4587 | 0.5277 | 0.5588 | 0.5565 | 0.05953 | 0.484 → 0.103 | 0.4141 → 0.6875 |
| control, seed 43 | 0.4663 | 0.5254 | 0.5368 | 0.5368 | 0.08392 | 0.474 → 0.084 | 0.3809 → 0.6934 |

Wrong or incomplete diagnoses:
- calling the run healthy because `actor/kl_loss`, `actor/pg_clipfrac`, and `grad_norm` look normal;
- fixing or naming only `grouped_whiten` (`0.4807` / `0.4905 < 0.51`) or only `kl_penalty` (`0.4102` / `0.3654 < 0.51`);
- blaming `kl_loss_coef=0.001`, `lr=2.5e-6`, GSM8K reward extraction, question filtering, or W&B teardown errors.

## Reference solution

- **issue**: both bugs are required — `verl/utils/torch_functional.py::grouped_whiten` indexing whitened advantages with the forward sort permutation `order`, and `verl/trainer/ppo/core_algos.py::kl_penalty` routing `low_var_kl` into a sequence-squared straight-through backward term while leaving forward `actor/kl_loss` untouched. Naming only one gets partial `root_cause_score`.
- **solution**: restore per-UID standardization inside `compute_grpo_outcome_advantage` (removing `grouped_whiten`) or invert the permutation via out-of-place scatter, and restore `if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):` with per-token `backward_score = 0.5 * (logprob - ref_logprob).square()` in `kl_penalty` (`evaluations/solution/oracle/fix.diff`).

## Verification

- **issue**: an LLM judge scores `agent_answer.md` against the hidden oracle on verdict, root cause, and evidence (`0.2 / 0.6 / 0.2`). Passing requires `verdict_score == 1.0` and both other scores above `0.8`.
- **solution**: the agent edits `/workspace`, writes `report.md`, and does not run training. The harness reruns `run_grpo_gsm8k.sh` (50 steps, 2 h cap) per `run_spec.json`, and `grade_submission.py` re-scores `experiment_logs/val_generations/50.jsonl` (`last_number` rule, `>= 1,300` rows, pass at `>= 0.51`). Calibrated across 8 runs (2 defective `0.4610` / `0.4640`, 2 `kl_penalty`-only `0.4102` / `0.3654`, 2 `grouped_whiten`-only `0.4807` / `0.4905`, and 2 controls `0.5588` / `0.5368` from step-0 `0.4587` / `0.4663`): `+0.0195` above the highest non-control run and `-0.0268` below the lower control. Grader replay exits `1` on all 6 non-control runs and `0` on both controls.
