# GRPO advantage broadcast along the sequence axis

## Case overview

- Task ID: `verl_grpo_advantage_broadcast`
- Case family: `reward_design`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

`compute_grpo_outcome_advantage` returns shape `[batch, response_length]`, but
the per-sample advantages are written along the sequence axis and expanded down
every row before the response mask. Training does not crash; the policy
gradient is decoupled from each sample's reward and held-out accuracy does not
improve.

An earlier version also injected a scorer defect (`taco.py::_to_bool_results`
calling `.item(0)` on a possibly empty array). It was removed: it never fired in
the recorded run and could only be found by code reading. The scorer now treats
a size-0 comparison as a failed case.

## Workspace and hidden evaluations


| Path                    | Visibility and purpose                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `workspace/`            | Agent-visible snapshot: code, one recorded run (wandb + hydra outputs).                   |
| `evaluations/issue/`    | Neutral diagnosis instruction, issue oracle, and task-local issue verifier.                |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier.           |


The agent cannot read oracle or verifier paths. `issue` grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Recorded run

- `workspace/logs/wandb/offline-run-20260926_105216-l7my3znq/files/output.log`
  (console with per-step metrics), `files/config.yaml`, and
  `workspace/outputs/2026-09-26/10-48-56/.hydra/`.
- 4 GPUs, train batch 16 x `n=8`, `max_response_length=4096`, KL and entropy
  off. Configured for 400 steps; the log ends at step 284 after about 8 hours of
  step time with no exception (job time limit).
- `critic/rewards/mean` averages 0.122 (steps 1-284, none 0). Held-out
  `val-core/taco/acc/mean@1`: 0.0761 / 0.0489 / 0.0598 / 0.0707 / 0.0435 /
  0.0543 at steps 0 / 50 / 100 / 150 / 200 / 250.
- Fingerprint: the token-weighted `critic/advantages/mean` stays near zero
  (median |mean| 4.9e-4) versus 3.9e-3 to 9.3e-3 on the correct-code TACO runs in
  `task_train_too_hard_new`, `task_function_call`, and `torch_functional`.
- Sanitized: the original experiment name and the source machine's task path
  named the defect. They were rewritten to `taco_qwen2_5_1.5b_grpo` /
  `tasks/taco_grpo` in the text artifacts, and the binary `run-*.wandb` (which
  carries CRC-checked records) was removed. The original 254 MB console log was
  only an LFS pointer in this repo and was removed.
- The run is attributed to the defective code from its experiment name, paths,
  and the fingerprint above; the code at run time was not recorded.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause target:

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`

Required fix targets:

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`: `restore`; target state/value: `scores = scores.unsqueeze(-1) * response_mask`, so each row carries that sample's scalar GRPO advantage on its response tokens.

## Evaluation design

`issue` grades `$AUTORLBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass requires `verdict_score` == 1.0, and `root_cause_score` and `evidence_score` each > 0.8; the weighted total alone is not a pass condition.

`solution` does not grade a structured diagnosis. The agent repairs the staged workspace and submits `report.md` without
launching training. The harness runs the designated launch script for 400 steps
and produces fresh experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.10`
  (10%) scores `1`, otherwise `0`. The healthy reference on this data and
  launch (`torch_functional`, fixed) peaks at 0.125; the defective run here
  peaks at 0.076.
- training reward: the mean `critic/rewards/mean` over training steps must be
  strictly above `0.14` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

## External dataset asset

- Repository: `cyh2004/ptdbench-taco-verl-easy` (neutral name — does not encode the task)
- Dataset license: `Apache-2.0`
- Provenance: TACO EASY snapshot (8368 train / 184 test), same bytes as `task_function_call`.
- Contents: `train.parquet` and `test.parquet` flat at the repo root (no `tar.gz`).

| File            | Size              | SHA-256                                                            |
| --------------- | ----------------- | ------------------------------------------------------------------ |
| `test.parquet`  | 61,082,676 bytes  | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `train.parquet` | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |

`environment/prepare.sh` fetches the whole repo straight into
`workspace/taco_verl`; the two parquet files are what training reads. No unpack
or checksum step — the dataset is delivered already extracted.

### How the flat dataset was published

One-time, with an HF token, to the **neutral-named** repo above (so the repo /
cache dir name never reveals the task to the agent). Recorded here so it can be
reproduced if the repo is ever lost:

```bash
# extract the two parquets from the legacy tarball (or reuse an existing taco_verl)
hf download LIF1014/ptdbench-reward-design-task-grpo-broadcast-dataset \
    artifacts/workspace-assets.tar.gz --repo-type dataset --local-dir /tmp/taco-dl
mkdir -p /tmp/taco-flat
tar -xzf /tmp/taco-dl/artifacts/workspace-assets.tar.gz -C /tmp/taco-flat workspace/taco_verl

# publish the two parquets at the repo root
hf repo create cyh2004/ptdbench-taco-verl-easy --repo-type dataset
hf upload cyh2004/ptdbench-taco-verl-easy /tmp/taco-flat/workspace/taco_verl . --repo-type dataset
```

The SHA-256 / size rows above are kept for reference only; `prepare.sh` no
longer checks them.

## External model asset

- Repository: `Qwen/Qwen2.5-1.5B-Instruct` (official)
- Pinned revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Byte-identical, on every allowed file, to the former task snapshot
  `LIF1014/ptdbench-verl-implementation-torch-functional@36a4c22`; switched
  because that repo name is visible to the agent through the shared download cache.
- Model: `Qwen2.5-1.5B-Instruct` (`model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`)

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- The designated experiment uses 8 GPUs, a 400-step budget, and a large shared-memory allocation.
- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`


The grader requires the final validation point and every training-reward step
from 1 through the declared horizon; partial or over-budget logs score 0.
This completion gate does not change the existing metric thresholds.
