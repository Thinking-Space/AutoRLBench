# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

- `workspace/scoring/math_verify.py` (`_get_pool`, `compute_score`)
  - kind: `symbol`
  - observed: a scorer process lazily creates a singleton spawn-based ProcessPoolExecutor. A child exits unexpectedly and the pool becomes unusable. The singleton is retained, so later submissions keep reaching the same unusable executor. The generic exception handler prints an error and returns the default ordinary score of zero. The cause of the original child exit is unknown.
  - expected: make an infrastructure failure distinguishable from a valid negative answer, then either restore scoring safely or fail visibly and retain the same answers for regrading. The worker and the process pool are separate layers.

## Observable impact

- The PPO run continues its optimizer updates. The evaluator runs the same 100 AIME24, AIME25, and AMC23 questions with 32 samples per question. The affected worker logs 1,300 scorer errors in the step-500 validation window.
- The original step-500 response-weighted score is 924/3200 (28.875%). Regrading the same question, answer text, GT, step, and UID gives 1589/3200 (49.65625%). Exactly 665 labels change, all from 0 to 1. The reviewer-side `archive_comparison.json` records the counts and original digests. A diagnosis can establish scoring corruption from the available source, contextual log, and unchanged examples without knowing the private regrade total.
- For example, visible archive row `011efd22-77c8-4c6b-824f-89f38ce6e6c4_0_0` (AMC23) ends in `\boxed{5}`, has ground truth 5, and received score 0. Accept visible evidence of this kind with the worker logs and source trace in place of the full regrade totals.
- VeRL consumes the validation reward as its score/accuracy. Repeated zero returns therefore lower held-out metrics without a new policy generation or optimizer step. The three published suites are validation inputs in `ppo.yaml`, and `reward.py` routes them through MathVerify. The DAPO training data uses the distinct strict boxed scorer. Early initialization or process errors outside this window are not evidence of a model regression.

## Trigger conditions

The MathVerify child must exit abnormally while its worker is active. Follow-on evaluation requests handled by the same scorer process reuse the executor. Other workers can still return valid judgments. A normal wrong answer can still legitimately return zero.

## Documentation reconciliation

The archived run, fresh startup, and controlled same-response replay are separate. Do not attribute the whole training curve or the first child exit to the defect. A response-length theory, an isolated checkpoint-effect theory, or an explanation based only on overlapping evaluation examples cannot account for the unchanged texts whose scores flip upon regrading.
