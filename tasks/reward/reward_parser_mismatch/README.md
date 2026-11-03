# PPO-Critic run assessment: reward_parser_mismatch

## Case overview

- Evaluation key: `reward_parser_mismatch`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `GuruMath controlled smoke` (RL prompts) + AIME-32 / MATH-500-32 held-out
- Training framework: `verl` (VIMPO PPO-Critic recipe)
- Launch script: `run_main.sh` → `recipe/vimpo/run_vimpo_critic.sh`
- Primary source evaluation metric: `val-core/math__math/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

The staged workspace reproduces the VIMPO PPO-Critic recipe on an unmodified
verl snapshot, but a reward-parser contract mismatch has been injected through a
custom reward function. `run_main.sh` exports `SMOKE_SCORING=1`, under which the
recipe script sources `recipe/vimpo/config/scoring.env`; that file sets
`REWARD_OVERRIDE` carrying
`reward.custom_reward_function.path=${PROJECT_ROOT}/autorlbench_case/math_reward.py`,
which the recipe script places on the Hydra command line (superseding the YAML
default). That
function extracts the model answer with an XML-tag regex
(`<answer>\s*(.*?)\s*</answer>`), while the GuruMath prompts and ground truth use
boxed answers (`\boxed{...}`). No response ever matches, so every rollout scores
`0.0` and `grade_answer` is never reached. The reward signal is identically zero.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `workspace/autorlbench_case/` | Preserved controlled evidence: injected `math_reward.py` + decoy `gurumath_qa_em.py` + unwired sibling `gsm8k_exact_match.py` (all documented by a maintainer-unfaithful `README.md`), controlled assets, and `logs/observed_buggy_run.log` (the observed log itself is not shipped; its frozen values live in `historical_evidence_manifest.json`). |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local LLM-judge verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The public package does not contain the historical log referenced above:
`workspace/autorlbench_case/logs/observed_buggy_run.log` is absent in both this
checkout and the corresponding AutoRLBench task. The historical values below are
therefore provenance claims from the manifest; reproduce them with a fresh run
when local log evidence is required.

The agent cannot read oracle or verifier paths. `issue` grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM judge.
`solution` grading reconstructs `val-core/math__math/acc/mean@1` and
`critic/rewards/mean` from the logs the harness rerun writes under
`$AUTORLBENCH_RESULTS/experiment_logs`.

## Original failure evidence

The historical manifest reports 5 PPO steps (`test_freq=1`,
`val_before_train=True`):

- `critic/rewards/mean` and `critic/score/mean` are `0.0` at every step
- `val-aux/math__math/parser_matched/mean@1` is `0.0` (no response matched)
- `val-core/math__math/acc/mean@1` and `val-core/math__aime_repeated_8x/acc/mean@1`
  stay `0.0`, flat across every checkpoint including the pre-training one
- `timing_s/reward` is non-zero (~0.0002 s), so the reward function runs — it
  simply never matches
- All 5 PPO steps complete; nothing raises, crashes, or warns on the reward path
- Advantages carry no learning signal (near-zero noise), so the policy never
  moves and validation stays `0.0`

## Why the recipe is unhealthy

Apart from the reward parser, the recipe is reproduced correctly, so the defect
is isolated to the injected custom reward rather than the rollout or algorithm
config. The verl framework code under `workspace/verl/` and `workspace/recipe/`
is byte-for-byte identical to the healthy control; only the task-authored glue differs
(`run_main.sh`, the smoke-lane scoring selection in
`recipe/vimpo/run_vimpo_critic.sh` + `recipe/vimpo/config/scoring.env`, the
`custom_reward_function` default in `recipe/vimpo/config/vimpo_trainer.yaml`, and
`autorlbench_case/{math_reward,gurumath_qa_em,gsm8k_exact_match}.py` + `README.md`).

- **Reward defect** (`autorlbench_case/math_reward.py::compute_score`, the grader
  carried by `REWARD_OVERRIDE` from `recipe/vimpo/config/scoring.env` under the
  `SMOKE_SCORING=1` smoke-lane flag): the
  answer is extracted with `re.search(r"<answer>\s*(.*?)\s*</answer>", ...)`.
  GuruMath answers are boxed, so the match is always `None` and the function
  returns `{"score": 0.0, "acc": False, "parser_matched": False}` for every
  sample.

The surrounding recipe is otherwise correct — these are not the defect: the
VIMPO PPO-Critic algorithm/optimizer config (actor `LR=1e-6`, critic
`CRITIC_LR=1e-5`), the FSDP/vLLM wiring, and the controlled smoke downscaling
(4 GPUs, 5 steps, batch 16, 1024-token responses) are expected harness
properties, not causes.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the XML-tag answer parser in
  `autorlbench_case/math_reward.py::compute_score` (the grader carried by
  `REWARD_OVERRIDE` from `recipe/vimpo/config/scoring.env` under
  `SMOKE_SCORING=1`, not the YAML-declared `gurumath_qa_em.py` decoy).
- Required fix target: extract the boxed answer (`\boxed{...}`, last match) and
  grade it, keeping the `compute_score(data_source, solution_str, ground_truth,
  extra_info=None, **kwargs)` signature and `{score, acc, parser_matched}`
  return contract.

An agent that attributes the flat reward to the optimizer, policy collapse, or
an infrastructure-startup failure is wrong.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.

`solution` is deterministic and metric-only, in the same shape as
`infrastructure/torch_functional`: fix the reward, rerun the launch script
for real, and the verifier reconstructs the metric from the logs under
`$AUTORLBENCH_RESULTS/experiment_logs`. Scoring is validation-first
(`score_mode: final`): the terminal validation accuracy decides the outcome,
with the training-reward gate only as a fallback for runs that logged no
validation metric; the score is `1.0` on pass and `0.0` otherwise.
See `evaluations/solution/verifier/resources/README.md`.

## Environment

The task runs in the image `ppo-critic-vimpo`
(`docker/images/ppo-critic-vimpo.Dockerfile`): Python 3.12,
CUDA 12.8, torch 2.8.0, vLLM 0.11.0, Ray 2.56.0, transformers 4.57.6,
flash-attn 2.8.3.post1.
This stack is deliberately **not** the shared `verl` image (vLLM 0.17 / torch
2.10 / CUDA 12.9): the workspace ships its own veRL + VIMPO recipe written
against vLLM 0.11, and the recorded val-core / critic-reward baselines the
solution grader reconstructs were produced on this exact stack. The workspace
veRL is not baked into the image; the harness bind-mounts the workspace at
`/workspace` and `PYTHONPATH=/workspace` makes `import verl` resolve to it.
`environment/prepare.sh` fetches the whole `Qwen/Qwen3-4B-Base` repo to
`/workspace/Qwen3-4B-Base` (the `MODEL_PATH` the effective run uses, per
`solution/instruction.md`) via the task-local `environment/fetch_hf.py`
(cache-first under `$AUTORLBENCH_MODELS`; no revision or checksum pins — the
recorded runs used the main-branch snapshot as of 2026-08-18, and the fetcher
takes whatever the repo's default branch serves). The controlled
training and held-out assets are tiny and ship embedded in
`workspace/autorlbench_case/assets/*.parquet`, so prepare does not re-fetch them;
the dependency stack is baked into the shared image and the workspace veRL needs
no editable install (`import verl` resolves via `PYTHONPATH=/workspace`;
`__version__` comes from `verl/version/version`).

The recipe needs **4 GPUs** (`run_main.sh`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`N_GPUS_PER_NODE=4`); `MODEL_PATH` must point at a local Qwen3-4B-Base checkpoint
(a large external resource, not embedded).

## Task hardening (2026-11-03, maintainers only)

The original packaging self-incriminated (filename `buggy_reward_fn.py`, a
`# BUG:` comment, `run_buggy.sh`, `PROJECT_NAME=autorlbench-type2-reward-parser`,
and an agent-runnable one-command oracle), so 9/10 issue configs scored 1.0 in
the first evaluation round. Two hardening rounds followed, both keeping the
defect, replay behavior, and environment byte-equivalent:

- **v1 (re-camo)**: `buggy_reward_fn.py` → `math_reward.py` with an innocuous
  docstring, `run_buggy.sh` → `run_main.sh`, neutral `PROJECT_NAME`, the
  one-command oracle moved to maintainer-side `evaluations/solution/oracle/`,
  and a *correct* decoy grader (`gurumath_qa_em.py`) falsely declared as wired
  in `vimpo_trainer.yaml` while the recipe script's Hydra CLI override carried
  the live path. Verified outcome: astra **fails** (0.88, evidence 0.4
  against the >0.8 gate), fable still passes (0.996).
- **v2 (wiring indirection + dilution)**: the CLI override is no longer a
  literal in the recipe script. `run_main.sh` only exports `SMOKE_SCORING=1`;
  the recipe script conditionally sources `config/scoring.env`, whose
  `REWARD_OVERRIDE` carries the live path; a maintainer-unfaithful
  `autorlbench_case/README.md` presents `gurumath_qa_em.py` as the default scorer,
  and an unwired `gsm8k_exact_match.py` sibling dilutes the grader set. The
  shell expansion was verified argv-identical to v1 (see
  `evaluations/solution/oracle/local_stub_check.py` and `reproduce_bug.py`,
  which resolves the chain). Verified outcome: fable passes at
  0.971 but its evidence score drops to 0.857 — one item (the non-zero
  `timing_s/reward`, i.e. the reward function runs but never matches) short of
  the >0.8 gate; astra fails v1 at 0.88 (evidence 0.4).
- **v3 (evidence-completeness bar) → v3.1 recalibration**: the issue oracle's
  "Observable impact" record was expanded from 4 to 8 frozen items to raise the
  evidence-completeness bar under the judge's boolean-enumeration rubric.
  Verified outcome (2026-11-03): **both targets fail** — fable 0.89
  (verdict 1.0, root_cause 1.0, evidence 0.45) and astra 0.867
  (verdict 1.0, root_cause 1.0, evidence 0.333). Interim sweep data then showed
  two of the added items (healthy `actor/grad_norm`/timings, non-zero
  `response_length/mean`) were stated by **zero of five** judged configs — pure
  padding that depressed every score uniformly instead of separating good from
  bad diagnoses (the strongest model scored below mid-tier ones on evidence) —
  and the "silent completion" item, also zero-hit, was retained because it is
  the defining symptom of this failure class (an expert diagnosis of a silent
  defect states that no error is ever raised). v3.1 drops the two padding items
  and softens
  the advantages phrasing to "no learning signal (near-zero noise)" so a
  correct-direction summary counts; everything else — flat-across-checkpoints
  validation, parser-match metric, advantages, non-zero reward timing, silent
  five-step completion — stays. No environment, workspace, or wiring change at
  any step; solution grading is metric-only and unaffected.

  **v3.1 full sweep (2026-11-03, 10 configs, effort=max, no opus):** all ten
  fail the issue gate with `verdict_score` 1.0 and `root_cause_score` 1.0 —
  the difficulty is entirely in evidence completeness. Evidence scores
  (uniform local re-judge against the v3.1 oracle): fable/glm/kimi/qwen/grok
  0.800 (at the strict >0.8 gate), deepseek 0.700, gemini 0.667, astra 0.450,
  sol 0.400, kimi31 0.330. Two oracle items were stated by no config
  (non-zero `timing_s/reward`; explicit silent five-step completion with no
  reward-path error) — they are the pass margin. Judge-variance spot checks:
  fable 0.78–0.80 across four samples and glm/kimi/deepseek/grok never above
  0.80; qwen crossed once (0.818, 9/11) in three samples. Infra note: three
  early trials (glm/kimi31/gemini) hit in-trial judge read-timeouts and were
  re-judged locally per the standard rescue protocol.

  **Recorded outcome (2026-11-03):** the ten pre-hardening rows
  (nine 1.0 and one 0.94 against the self-incriminating packaging) were
  replaced with this sweep's results; the old opus row was dropped with no
  replacement per the standing no-opus publication policy. A fresh end-to-end
  stability pair on the final v3.1 task put fable exactly at the boundary:
  one trial wrote the complete evidence record and passed (evidence 0.90,
  weighted 0.98 — the recorded row), the paired trial scored 0.80 (fail). So
  the honest statement is that v3.1 fails nine configs and holds fable at a
  coin-flip on the evidence-completeness gate, not a clean ten-way fail.

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.
