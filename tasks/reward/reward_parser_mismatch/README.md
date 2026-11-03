# PPO-Critic run assessment: reward_parser_mismatch

## Case overview

- Evaluation key: `reward_parser_mismatch`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `GuruMath controlled smoke` (RL prompts) + AIME-32 / MATH-500-32 held-out
- Training framework: `verl` (VIMPO PPO-Critic recipe)
- Launch script: `run_buggy.sh` → `recipe/vimpo/run_vimpo_critic.sh`
- Primary source evaluation metric: `val-core/math__math/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

The staged workspace reproduces the VIMPO PPO-Critic recipe on an unmodified
verl snapshot, but a reward-parser contract mismatch has been injected through a
custom reward function. `run_buggy.sh` wires
`reward.custom_reward_function.path=autorlbench_case/buggy_reward_fn.py`. That
function extracts the model answer with an XML-tag regex
(`<answer>\s*(.*?)\s*</answer>`), while the GuruMath prompts and ground truth use
boxed answers (`\boxed{...}`). No response ever matches, so every rollout scores
`0.0` and `grade_answer` is never reached. The reward signal is identically zero.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `workspace/autorlbench_case/` | Preserved controlled evidence: injected `buggy_reward_fn.py` and controlled assets. The historical log named by the manifest is not included. |
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
  stay `0.0`
- `timing_s/reward` is non-zero (~0.0002 s), so the reward function runs — it
  simply never matches

PPO initializes and completes all steps; gradients, critic updates, and MFU look
healthy. The defect is silent: no exception is raised, but advantages are noise
around zero and the policy learns nothing.

## Why the recipe is unhealthy

Apart from the reward parser, the recipe is reproduced correctly, so the defect
is isolated to the injected custom reward rather than the rollout or algorithm
config. The verl framework code under `workspace/verl/` and `workspace/recipe/`
is byte-for-byte identical to the healthy control; only `run_buggy.sh` and the
new `autorlbench_case/buggy_reward_fn.py` differ.

- **Reward defect** (`autorlbench_case/buggy_reward_fn.py::compute_score`): the
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
  `autorlbench_case/buggy_reward_fn.py::compute_score`.
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
`$AUTORLBENCH_RESULTS/experiment_logs`. Score is
`0.8 * validation + 0.2 * training_reward`; pass requires both components
(`score == 1.0`). See `evaluations/solution/verifier/resources/README.md`.

## Environment

The task runs in the image `vimpo-cu128-recorded` (image not yet published;
definition `docker/images/vimpo-cu128-recorded.Dockerfile` is pending): Python 3.12,
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

The recipe needs **4 GPUs** (`run_buggy.sh`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`N_GPUS_PER_NODE=4`); `MODEL_PATH` must point at a local Qwen3-4B-Base checkpoint
(a large external resource, not embedded).

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.

The grader requires the final validation point and every training-reward step
from 1 through the declared horizon; partial or over-budget logs score 0.
This completion gate does not change the existing metric thresholds.
