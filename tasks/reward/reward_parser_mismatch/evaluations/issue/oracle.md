# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `autorlbench_case/math_reward.py::compute_score`
  - kind: `symbol`
  - observed: answer extracted with `re.search(r"<answer>\s*(.*?)\s*</answer>", solution_str, re.DOTALL)`; on no match returns `{"score": 0.0, "acc": False, "parser_matched": False}`
  - expected: extract the boxed answer (`\boxed{...}`, last match) and grade it, so correct boxed answers receive reward
  - note: this reward is the one actually wired into training (see wiring below). The GuruMath prompts and ground truth (train and held-out parquets) uniformly instruct `Please output the final answer within \boxed{}`, which the XML-tag regex never matches.

## Wiring (which grader is actually live)

- `run_main.sh` exports `SMOKE_SCORING=1` (the smoke-budget lane) and invokes
  `recipe/vimpo/run_vimpo_critic.sh`. Under that flag the recipe script sources
  `recipe/vimpo/config/scoring.env`, which sets
  `REWARD_OVERRIDE="reward.custom_reward_function.path=${PROJECT_ROOT}/autorlbench_case/math_reward.py
  reward.custom_reward_function.name=compute_score"`; the
  `python3 -m recipe.vimpo.main_vimpo` command line carries `${REWARD_OVERRIDE}`
  (unquoted, so it expands to the two Hydra overrides).
- `recipe/vimpo/config/vimpo_trainer.yaml` also sets
  `reward.custom_reward_function.path: autorlbench_case/gurumath_qa_em.py`, but the
  Hydra command-line override supersedes the YAML value, so `gurumath_qa_em.py`
  (a boxed-extraction grader that would score the run correctly) is **not** in
  effect. Diagnoses that treat the YAML as the effective wiring, or that blame
  `gurumath_qa_em.py`, are misdiagnoses.

## Observable impact

- `critic/rewards/mean` and `critic/score/mean` are `0.0` at every step
- `val-aux/math__math/parser_matched/mean@1` is `0.0` (no response matched)
- `val-core/math__math/acc/mean@1` and `val-core/math__aime_repeated_8x/acc/mean@1` stay `0.0`, flat across every checkpoint including the pre-training one
- `timing_s/reward` is non-zero, so the reward function runs but never matches
- All 5 PPO steps complete and nothing raises, crashes, or warns on the reward path — the defect is silent
- Advantages carry no learning signal (near-zero noise), so the policy never moves and validation stays `0.0`

## Trigger conditions

- The `<answer>...</answer>` parser in the live custom reward cannot match the
  boxed answers the prompts request.

## Documentation reconciliation

- The docstring of `math_reward.py` frames the answer-tag extraction as the
  shared QA-contract pipeline (`search_r1_like_qa_em`); the shipped parquet
  prompts contradict it: this run's data asks for `\boxed{}`. Comments are not
  evidence — the effective configuration plus the data are.
- The comment block in `recipe/vimpo/config/scoring.env` and the module list in
  `autorlbench_case/README.md` frame `math_reward.py` as a comparable-variant /
  QA-tag-lane scorer and present `gurumath_qa_em.py` as the default; the
  effective wiring (the `${REWARD_OVERRIDE}` Hydra overrides) contradicts both.
  Prose documentation is not evidence — trace the launch chain.
- PPO continuing to execute normally (finite loss, healthy-looking gradients
  and MFU) is a symptom of the silent zero reward, not evidence of health.
  This is a reward-parser contract mismatch, not an optimizer,
  policy-collapse, or startup failure. The smoke-budget settings (5 steps,
  batch 16, 1024-token responses, `LR=1e-6`) are harness properties, not causes.
