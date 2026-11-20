# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

In `workspace/qat/vendor/verl/verl/trainer/ppo/ray_trainer.py`, the active native rollout path keeps the student top-k IDs and log probabilities emitted by vLLM. In `workspace/qat/vendor/verl/verl/workers/actor/dp_actor.py::compute_distillation_reward`, the student scores on teacher-selected IDs, `S_on_T`, come from the HF actor's `_forward_micro_batch` on the same response prefix. The active `union` branch concatenates those two sources as `S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)`. It masks the duplicate teacher IDs, then uses the concatenation for the student–teacher log-ratio and for reward weighting. The optimization forward pass later recomputes actor scores for the union IDs through the training model.

- Observed — In a single response position, which model selected an ID determines the runtime used to score that ID under the student. Student-selected IDs use the quantized rollout path. Teacher-only IDs use the separate learner forward pass. Normalizing the concatenated values does not reconcile the two distributions.
- Expected — Use a common student scoring path for all IDs in the reward's candidate union. The original inference probabilities can still describe the behavior policy that generated the trajectory. Keep the candidate IDs, teacher scores and duplicate mask fixed when testing the change, and preserve the configured PPO ratio and block-alignment term.

## Observable impact and verification

The two-GPU run improves from 31/240 before training to 62/240 at step 16, falls to 54/240 by step 30 and to 36/240 at step 60. All 240 held-out answers finish naturally at both late checkpoints. Over steps 31–60, the logged perplexity falls from 1.98 to 1.20, the rollout-to-learner logprob gap rises from 0.362 to 1.388, and the gradient norm rises from 2.36 to 10.46. The native rollout path feeds its own sampled token probabilities into the metric logged as `training_ppl`. This measures confidence on sampled answers, not held-out correctness. The original four-GPU incident showed the same direction, falling from 238/960 at step 30 to 133/960 at step 60.

In the author-only same-response comparison, replacing only the student scoring source at fixed candidate IDs changes the sum over candidate rewards by a mean absolute 0.2395 per response token over 221,335 tokens, and 357 positions cross zero. After a common step-16 checkpoint, the two arms each reach 54/240 at step 30; at step 60 the faulty arm reaches 36/240 and the repaired arm reaches 63/240. The 11.25-point endpoint gap has a question-bootstrap interval of +4.17 to +19.17 points. Author-only numbers are not required in the diagnosis.

## Grading notes

Full root-cause credit requires both score origins, the union operation, and the effect on the reward. A diagnosis that finds rollout/learner probability disagreement without identifying the two sources within the same reward vector earns partial root-cause credit. Credit a diagnosis that proposes a consistent scoring source for the union even if it chooses a different consistent source from the authors' repair.

Training output truncation can affect learning and occurs in this run. A diagnosis attributing the entire defect to an 8K generation limit, held-out truncation, a wrong grader, reference row reordering, or the intended use of behavior log probabilities does not explain the mixed reward computation. Claims that the model never learned conflict with the early held-out checks. The response need not claim that the mixed scores account for the entire held-out decline.
