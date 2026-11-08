# GSM8K reward credits the gold number anywhere in the response

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K with a hand-written reward that gives full credit whenever the gold answer appears among the numbers in the response. The training score and the logged validation accuracy climb for 100 steps, but the policy never has to commit to a final answer: scored strictly, held-out accuracy is 0, and scored format-free it falls from 0.46 to 0.13.

| | |
| --- | --- |
| **Author** | Ziqi Zhou |
| **Affiliation** | Duke University |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | `reward-hacking` `lenient-verifier` `answer-extraction` |
| **Framework** | verl v0.9.0 (`483b8a0`), FSDP + vLLM 0.28 colocated, custom reward `rewards/gsm8k_reward.py` |
| **Run time** | ~2.7 hours per run (control ≈ 2.1 h) |
| **Resources** | 1 GPU (recorded on 1 × RTX A6000 48 GB) |

## Difficulty

The recipe is verl's GSM8K GRPO quickstart scaled to one GPU, with the reward swapped for the
user's own `rewards/gsm8k_reward.py` through `reward.custom_reward_function`. Its docstring gives
the reason every practitioner recognises: verl's built-in rule only accepts `#### <number>` and
scored 0.0008 at step 0, "so the policy saw no signal". The replacement extracts every number in the
response and returns 1.0 if the normalised gold answer is among them. verl uses the same function
for `val-core/openai/gsm8k/acc/mean@1`.

The function is wrong in a way that optimisation finds immediately. At step 0 it already pays
0.594 on the held-out set and 0.54 on sampled rollouts, although only 0.1% of those responses state
the answer as asked and 46% get it right under any reading of "final answer": in 14% of responses
the gold number appears somewhere (a restated quantity, an intermediate result, one of several
candidates) while the last number written is wrong. GRPO rewards whatever raises that probability.
Over 100 steps the policy learns to keep enumerating: distinct numbers per response 9.4 → 14.3,
responses 306 → 431 tokens (43% hit the 512 limit at step 100, up from 10%), numbered "steps" that
recompute the same quantity or chain further arithmetic until cut off, and no `####` line at all
from step 40 (nothing pays for one). The logged score rises 0.54 → 0.81 and the logged validation
"accuracy" 0.594 → 0.684. Re-scored with a committed-answer rule (last `#### N`, the format the
prompt asks for) the same validation dumps give 0.000 from step 10 on. Scored format-free (last
number in the response) they fall 0.462 → 0.429 (30) → 0.339 (50) → 0.130 (100). The share of
rollouts that earn credit without a correct last number grows 0.14 → 0.59. The control differs only
in the reward file (last `#### N` equals gold): 0.557 strict at step 100, `####` in 99% of
responses, 240-token answers.

This is the "loose extraction" / "missing marker" / "contradiction blindness" family of verifier
bugs that Ray's verifier-fuzzing study lists as what practitioners ship (arXiv:2606.01066, Table 1),
with the predicted outcome: high reward, low strict correctness. Varambally et al. (arXiv:2604.15149)
show the same shortcut dynamics for extensional verifiers. The `reward-doctor` linter calls the
exploit a "shotgun" completion. The bug is not in verl: it is forty lines of user code, the first
thing one writes when the strict reward "doesn't work".

| step | 0 | 10 | 30 | 50 | 80 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| defective logged `val-core/openai/gsm8k/acc/mean@1` (its own rule) | 0.594 | 0.617 | 0.656 | 0.679 | 0.689 | 0.684 |
| defective strict (last `#### N`, independent re-score) | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| defective format-free (last number) | 0.462 | 0.454 | 0.429 | 0.339 | 0.244 | 0.130 |
| control logged = strict | 0.001 | 0.506 | 0.563 | 0.563 | 0.569 | 0.557 |
| control format-free (last number) | 0.462 | 0.503 | 0.562 | 0.564 | 0.569 | 0.557 |
| defective `critic/score/mean` (train) | n/a | 0.655 | 0.802 | 0.734 | 0.770 | 0.810 |
| defective hedged credit in rollouts (gold somewhere, last number wrong) | n/a | 0.166 | 0.205 | 0.298 | 0.486 | 0.589 |
| defective distinct numbers per rollout | n/a | 9.8 | 10.0 | 11.3 | 13.2 | 14.3 |

The defective run is not a failure to optimise. Advantages, gradient norms (0.56 to 0.77 vs the
control's 0.29 to 1.01), entropy (0.55 → 0.07 in both) and KL all look like the control's. Its
format-free accuracy on training rollouts even rises first (0.41 → 0.60 at step 30, the model does
get better at arithmetic while hedging is still cheap) before the exploit takes over and drives it to
0.22 at step 100.

Wrong diagnoses a reviewer should expect and reject:

- **"The run is healthy: reward and validation accuracy go up."** Both are computed by the
  defective rule. The frozen `val_generations/*.jsonl` let anyone re-score them: at step 100 none
  of the 1,319 validation responses ends with `#### <gold>`, and 87% do not even end on the gold
  number.
- **"The model never learned the `####` format, so it is a sparse-reward / capacity problem."** The
  control, with the same model, data and budget, writes `####` in 97.5% of validation responses by
  step 10 and reaches 0.557. The defective run has no reason to learn it: the reward pays without it.
- **Learning rate, KL, entropy, step budget.** Shared with the control. The defective run's
  optimisation is healthy. It optimises the wrong target.
- **"The reward is fine, it is just `flexible` scoring."** verl's `flexible` mode takes the *last*
  number. This rule takes *any* number. The difference is exactly the 14% hedged credit at step 0 and
  the 59% at step 100.
- **verl's reward manager / `custom_reward_function` plumbing.** It does what it is told. Every
  logged score equals `compute_score` applied to the dumped response.

## Reference solution

- **issue**: name the any-number match in `rewards/gsm8k_reward.py::compute_score` (used for both
  training and validation) as the reason the logged scores rise while the policy does not learn to
  answer: credit without a committed final answer, exploited by enumerating more numbers per
  response. "The model is not learning the format" without the reward's leniency is a symptom.
  "healthy", LR/KL/capacity/data and `flexible`-mode explanations fail.
- **solution**: credit only a committed final answer: the last `#### <number>` equals the gold
  answer (`evaluations/solution/oracle/fix.diff`, equivalent to deleting the custom reward and using
  verl's built-in strict rule). Any reward that requires one final answer in the requested format
  passes. A reward that keeps paying for the number anywhere does not, because the grader does not
  read the logged metric.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. A run passes only if the verdict score is 1.0 and the root-cause and evidence scores are both above 0.8. The weighted score is reported but does not decide the pass.
- **solution**: the agent submits `report.md` and does not train. The harness reruns the
  experiment per `run_spec.json` (`run_gsm8k_grpo.sh`, 100 steps, 4 h budget, offline-pinned env:
  WANDB offline mode and `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`/`HF_DATASETS_OFFLINE` set so the
  rerun stays inside the prepared image) and grades only that
  run. Because the defect is the metric itself, the grader ignores logged numbers and re-scores the
  run's own `experiment_logs/val_generations/100.jsonl` with a fixed strict rule (last `#### N`
  numerically equal to the gold answer, ≥ 1,300 rows). The accuracy must be ≥ 0.28. The defective
  run is 0.000 at step 100 and 0.0015 at its best (step 0) under that rule. The control is 0.557 at
  step 100 and 0.0008 at step 0, so applying the fix without the rerun fails. Calibrated on one run
  per side, so the margins (0.279 / 0.277) are not a variance estimate. The rerun is anchored:
  `run_gsm8k_grpo.sh` is byte-compared against the verifier's pristine copy and the GSM8K parquets
  are pinned by sha256 at grading time, and `metrics.jsonl` is required as run-reality evidence
  (one row per training step 0-100; values never scored), so a forged dump, an edited runner, a
  shortened run or swapped data fails. Grader replay on the recorded evidence (dump plus the run's
  `metrics.jsonl`): defective exit 1, control exit 0. A symlinked or missing dump exits 1. Not
  verified: the `verl-0.9-vllm0.28` image is not built and `prepare.sh` has not been run in Docker.
  both recorded runs used the same stack natively.
