# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$AUTORLBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$AUTORLBENCH_RESULTS/experiment_logs/wandb`.

## Which space `eval/aime` is read in

The math_dapo **score** space, `[-1, 1]`, exactly as `slime_retool` grades the
same recipe — and exactly the space the frozen historical trace was logged in
(`eval_reward_key: score` in `workspace/wandb/latest-run/files/config.yaml`).
No `--eval-reward-key` is involved.

That choice is what makes this level gradable at all, and it is the whole
discriminator:

- `reward_func`'s score channel is where the injected defect lives, so a run
  that still carries `5 * length_ratio` reports an `eval/aime` **outside**
  `[-1, 1]` — the historical defective run logs `+0.822` at step 0 rising to
  `+3.841` at step 99.
- A run with the length term removed reports the correctness signal, which
  cannot exceed `+1` and in practice sits well below `0`.

So "did the agent remove the length bonus" is read directly off the metric,
rather than inferred from a second-order accuracy collapse.

## Bounds and how they are calibrated

The reference is the repaired run at
`devspace 1177503:/user/chenyinghao/harbor-runs/manual-slime-retool` — the
canonical recipe with no length term, same image, same assets, same
`--num-rollout 70` budget, 8 eval points:

| step | 0 | 9 | 19 | 29 | 39 | 49 | 59 | 69 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `eval/aime` | -0.5808 | -0.5706 | -0.5078 | -0.5035 | -0.5065 | -0.4373 | **-0.4246** | -0.4861 |
| eval `truncated_ratio` | 0.008 | 0.006 | 0.008 | 0.008 | 0.019 | 0.056 | 0.079 | 0.125 |

`score_mode` is `best`: the verifier takes the highest finite `eval/aime` over
the run rather than the value at step 69. The bar here is "the run actually
trained", and a single eval point over 30 AIME problems is noisy enough that
the last one is a poor proxy for that.

- **Lower bound `-0.55`.** An untrained step 0 sits at `-0.5808` and fails.
  Because `score_mode` is `best`, early points still under the bar (step 9 is
  `-0.5706`) do not sink a run that later learns. The repaired run's best point
  is step 59 at `-0.4246`, `0.125` clear. Same bound as `slime_retool`, so the
  healthy twin and the repaired defect twin are judged against one curve.
- **Upper bound `0.5`.** Every eval point of a run that still has the length
  bonus is above it. The thinnest case is the very first eval, before length
  has inflated: the repaired step 0 is `-0.5808` and the step-0 length bonus is
  `5 * 4042 / 16384 = +1.233`, so an unrepaired step 0 lands near `+0.65` —
  `0.15` above the bound, and every later point is `+1.3` or more above it.
  In the other direction `0.5` is `0.92` above the repaired run's best point;
  clearing it legitimately would take roughly 75% AIME accuracy, which this
  recipe is nowhere near in 70 steps. The bound therefore cannot fail a correct
  repair.

Verified against both real traces: the repaired log scores `1`
(best `-0.4246` at step 59); the historical defective trace scores `0`
(`score 3.8410780588785807 at step 99 is above required maximum 0.5`). A
single-eval unrepaired run also scores `0` (`0.6527 at step 0`), and a
repaired run truncated at step 0 scores `0` (`-0.5808` below `-0.55`).

## Known limit

A partial repair that merely shrinks the coefficient (say `5` to `1`) can land
inside `[-0.55, 0.5]` and pass. Metric-only grading cannot separate that from a
real removal; `evaluations/issue` is what grades the attribution, and
`instruction.md` asks for the term to be removed. Do not lower the upper bound
to chase this — the margin against a correct repair is what protects the level
from false negatives.

## Accuracy is deliberately not graded

`reward_func` floors an incorrect sample at `min(-0.6, …)` rather than `-1`, so
a mean score of `-0.4861` does **not** convert to `(score + 1) / 2 = 25.7%`
accuracy; it only bounds accuracy to `[7.1%, 25.7%]`. No measured accuracy
exists for the repaired recipe, so no accuracy threshold can be calibrated.
Grade the score channel, which is measured.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
