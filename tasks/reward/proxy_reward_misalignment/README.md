# PPO-Critic run assessment: proxy_reward_misalignment

## Case overview

- Evaluation key: `proxy_reward_misalignment`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `text-complete fixed-16 v3` (16 RL prompts repeated 64 times) + AIME-32 / MATH-500-32 held-out
- Training framework: `verl` (VIMPO PPO-Critic recipe)
- Launch script: `run_buggy.sh` → `recipe/vimpo/run_vimpo_critic.sh`
- Primary source evaluation metric: `val-core/math__math/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

The active training data is **proxy-data-v3**. It repairs incomplete questions
while preserving 16 questions × 64 repetitions, row order, answers, training
budget and the injected reward defect. Seven cases are unchanged, seven have
explicit geometry/rounding conditions, and two are newly authored self-contained
replacements. The original data is archived under
`evaluations/solution/oracle/data_versions/v2/`; these are versioned task changes,
not edits to historical run records. The 16 answers have independent formula or
enumeration checks in `data_versions/rebuild_v3.py`.

Historical solution runs used v2. The numerical thresholds below remain the
historical values; paired GPU qualification is still needed for v3. The new
reward behavior gate has been verified on CPU. This does not claim an end-to-end
GPU run on the revised questions.

The staged workspace reproduces the VIMPO PPO-Critic recipe on an unmodified
verl snapshot, but a misspecified **proxy reward** has been injected through a
custom reward function. `run_buggy.sh` wires
`reward.custom_reward_function.path=autorlbench_case/type3_length_reward_v2_fn.py`.
For the controlled training source (`data_source == "autorlbench_type3_proxy_train_v2"`)
that function returns a **brevity proxy** as the training score:

```text
proxy_reward = max(0.0, min(1.0, 1.0 - n_chars / 4000.0))
```

Shorter responses receive higher reward regardless of mathematical correctness.
Validation (`data_source != TRAIN_SOURCE`) still scores genuine correctness, so
the training objective is misaligned with the target task. This is a
reward-hacking / proxy-objective-misalignment pattern, not a dead reward signal.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `workspace/autorlbench_case/` | Preserved controlled evidence: injected `type3_length_reward_v2_fn.py` and controlled assets. The historical log named by the manifest is not included. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local LLM-judge verifier. |
| `evaluations/solution/` | Repair-and-run instruction, versioned data/reference material, reward behavior checks and fresh-run metric verifier. |

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

For **v2 data**, the historical manifest reports 15 PPO steps (`test_freq=5`,
`val_before_train=True`):

- `critic/rewards/mean` (the proxy) is **high and rising**: `~0.52 → 0.67`
  across steps, mean `~0.54`.
- Held-out `val-core/math__math/acc/mean@1` **declines**: `0.78125 (step 0)` →
  `0.6875 (step 5)` → `0.65625 (step 10)` → `0.59375 (step 15)`.
- `response_length/mean` falls (~`677 → 474`): the policy is learning to be
  shorter, exactly what the proxy rewards.
- Auxiliary truth tracking (`true_accuracy/score_mean`) is ~`0.52` and does not
  improve, confirming the proxy gain does not reflect correctness.

PPO initializes and completes all 15 steps; gradients, critic updates, and MFU
look healthy. The defect is in the *objective*, not the optimizer.

## Why the recipe is unhealthy

The verl framework code under `workspace/verl/` and `workspace/recipe/` is
byte-for-byte identical to the healthy control; only `run_buggy.sh` and the
injected `autorlbench_case/type3_length_reward_v2_fn.py` differ.

- **Reward defect** (`autorlbench_case/type3_length_reward_v2_fn.py::compute_score`):
  for the training source the returned `score` is `1 - n_chars/4000` (brevity),
  not correctness. The function even carries truthful `acc` /
  `true_math_score` fields, so the misalignment is observable: reward goes up
  while `true_math_score` and held-out accuracy do not.

The surrounding recipe is otherwise correct — these are not the defect: the
VIMPO PPO-Critic algorithm/optimizer config (actor `LR=5e-6`, critic
`CRITIC_LR=1e-5`), the FSDP/vLLM wiring, and the controlled smoke downscaling
(4 GPUs, 15 steps, fixed-16 train set repeated x64, 1024-token responses) are
expected harness properties, not causes.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the brevity proxy in
  `autorlbench_case/type3_length_reward_v2_fn.py::compute_score` for the training
  source.
- Required fix target: score the training source by mathematical correctness
  (a short incorrect answer must score 0; both short and long correct boxed
  answers must score 1; response length must not determine the training
  reward), keeping the `compute_score(data_source, solution_str, ground_truth,
  extra_info=None, **kwargs)` signature. `oracle/fixed_reward_fn.py` routes the
  training source through `default_compute_score` on `math__math`.

An agent that attributes the symptom to a dead reward, a parser mismatch, an
actor-LR collapse, or a startup failure is wrong.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.

`solution` is rule-based: after the harness rerun, the verifier first checks
data/scorer hashes and 48 reward examples covering training/validation routing,
correct/wrong answers, short/long responses, nested boxes, empty answers and
last-box selection. A failure scores zero. It then reconstructs fresh metrics
under `$AUTORLBENCH_RESULTS/experiment_logs`. The historical numerical score is
`0.8 * validation + 0.2 * training_reward`; pass requires both components
(`score == 1.0`). See `evaluations/solution/verifier/resources/README.md`.

The reward checks prevent constant rewards and remaining length dependence;
they do not make arbitrary logs or submitted code trustworthy. The parent
checks expected results outside a child worker, but actual OS/container
isolation and independent protected model evaluation are still required for
hostile submissions. Thresholds for v3 require its own paired runs.

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
training set (`train_proxy_fixed16_x64_v3.parquet`) and held-out assets
(`aime32.parquet`, `math500_32.parquet`) are tiny and ship embedded in
`workspace/autorlbench_case/assets/`, so prepare does not re-fetch them; the
dependency stack is baked into the shared image and the workspace veRL needs no
editable install (`import verl` resolves via `PYTHONPATH=/workspace`;
`__version__` comes from `verl/version/version`).

The recipe needs **4 GPUs** (`run_buggy.sh`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`N_GPUS_PER_NODE=4`); `MODEL_PATH` must point at a local Qwen3-4B-Base checkpoint
(a large external resource, not embedded).

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.

The grader requires the final validation point and every training-reward step
from 1 through the declared horizon; partial or over-budget logs score 0.
This completion gate does not change the existing metric thresholds.

The auxiliary AIME file contains 4 unique questions repeated 8 times, whereas
the primary MATH validation set contains 32 unique questions. Counts of rows
must not be reported as counts of independent questions. Reproduce the active
data/answer/hash checks with:

```bash
python3 -B tasks/reward/proxy_reward_misalignment/evaluations/solution/oracle/data_versions/rebuild_v3.py
python3 -B audit/test_proxy_v3.py
```
