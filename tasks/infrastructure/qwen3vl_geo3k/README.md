# verl_implementation diagnosis: qwen3vl_geo3k

## Case overview

- Domain: infrastructure
- Evaluation key: `infrastructure/qwen3vl_geo3k`
- Case family: `infrastructure`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-VL-8B-Instruct`
- Dataset: `hiyouga/geometry3k` (2101 train / 601 held-out examples)
- Training framework: `verl` v0.9.0, FSDP2 + vLLM, GRPO
- Environment: `verl-qwen3vl-cu130` · 8 GPUs
- Primary diagnostic metric: frozen-input response-token log-probability disagreement (nat)
- Supporting training metric: `rollout_corr/kl` (nat/token)

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. The run completes and learns, but the training and
inference paths disagree on multimodal position semantics.

## Canonical failure reason

`verl/workers/config/model.py::HFModelConfig.__post_init__` constructs the
processor with raw `AutoProcessor.from_pretrained` instead of verl's
`hf_processor` factory. The raw processor lacks the dynamically bound
`get_rope_index` capability. The existing agent-loop fallback then assigns
flat positions to image sequences, while vLLM independently constructs the
correct multimodal positions. The actor and rollout engine therefore assign
different log-probabilities to the same response tokens.

Relative to the pinned upstream commit, the workspace replaces the processor
factory in `HFModelConfig.__post_init__`. Restoring the upstream factory is the
reference repair.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible defective source, launch script, and frozen experiment evidence. |
| `evaluations/issue/` | Neutral diagnosis instruction, hidden issue oracle, and standard judge. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and deterministic verifier. |
| `environment/` | Pinned asset preparation scripts; no repair implementation. |

The agent cannot read oracle or verifier paths. Issue grading compares
`$AUTORLBENCH_RESULTS/agent_answer.md` with the hidden oracle. Solution grading
requires a fresh run and fresh parity measurements. The diagnosis instruction and issue judge use the shared evaluation protocol.

## Source and experiment configuration

```yaml
source_framework: verl
upstream_commit: 483b8a009ba3a97563edee3a19887e4862b8094a
upstream_version: v0.9.0
evidence_kind: processor_initialization_regression
target_file: verl/workers/config/model.py
recorded_runs_per_side: 1
verify:
  num_gpus: 8
  gpu_model: H800
  max_train_steps: 20
  validation_interval: 5
  prompt_length: 2048
  max_response_length: 8192
  max_model_length: 10240
  token_budget: 16384
  train_batch_size: 32
  rollout_n: 5
  learning_rate: 1.0e-6
  kl_loss_coef: 0.01
```

Both recorded runs completed all 20 steps, with finite gradients, offline
W&B logs, and validation at steps 0, 5, 10, 15, and 20. Training averages below
are simple averages over steps 1–20. The frozen parity measurement uses six
cases and 64 response tokens per case before training; it is not a measurement
of the step-20 checkpoint.

| Recorded metric | Defective | Control |
| --- | ---: | ---: |
| Mean training–rollout KL (`rollout_corr/kl`), nat/token | 0.006591 | 0.000623 |
| Mean token-probability Pearson correlation | 0.993060 | 0.999253 |
| Mean training reward | 0.548156 | 0.545062 |
| Mean gradient norm | 0.410747 | 0.462388 |
| Mean response length, tokens | 2476.38 | 2425.43 |
| Frozen maximum absolute token log-probability difference, nat | 3.740287 | 0.550462 |
| Mean training step time, seconds | 118.49 | 116.43 |

| Validation step | Defective correct / 601 | Control correct / 601 | Defective binary accuracy | Control binary accuracy | Defective logged `acc` | Control logged `acc` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 343 (57.07%) | 353 (58.74%) | 57.07% | 58.74% | 0.513644 | 0.528619 |
| 5 | 368 (61.23%) | 363 (60.40%) | 61.23% | 60.40% | 0.551082 | 0.543594 |
| 10 | 350 (58.24%) | 356 (59.23%) | 58.24% | 59.23% | 0.524126 | 0.533111 |
| 15 | 353 (58.74%) | 367 (61.06%) | 58.74% | 61.06% | 0.528619 | 0.549584 |
| 20 | 355 (59.07%) | 370 (61.56%) | 59.07% | 61.56% | 0.531614 | 0.554077 |

Geo3K reward is `0.9 * correctness + 0.1 * format`. Recorded validation outputs
have rewards 0 or 0.9; the log's `acc` field also carries this reward. Step-20
raw `acc` is 0.531614 / 0.554077, whereas the table reports binary correctness
from validation records. `training/rollout_probs_diff_*` measures probability
differences, not nats. `actor/kl_loss` compares against the reference policy,
not the rollout engine.

There is one recorded run per side. Initial validation already differs between
the two runs, so the final accuracy gap is not entirely attributable to the
repair. This short experiment establishes persistent training–inference
inconsistency; it does not establish training collapse or multi-seed variance.
All per-step scalar metrics are preserved in `workspace/logs/recorded_run/`;
the hidden oracle contains the control logs and comparison CSVs.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause target:

- `verl/workers/config/model.py::HFModelConfig.__post_init__`

Required fix: restore processor initialization that retains verl's multimodal
position capability. The reference implementation calls `hf_processor` and
preserves `get_rope_index`. Keep the pinned model, dataset, reward, launch
budget, and validation protocol unchanged.

Valid shapes, finite gradients, plausible reward, and improving validation do
not rule out this defect. Attributing the failure solely to BF16, GRPO, reward,
or an absence of learning is incorrect.

## Evaluation design

`issue` uses the standard LLM judge: verdict / root cause / evidence weights
are 0.2 / 0.6 / 0.2; passing requires `verdict_score` == 1.0 with `root_cause_score` and `evidence_score` each > 0.8, not just a weighted total of at least 0.8.

`solution` requires a fresh foreground 20-step training run, finite nonzero
gradients, repeated validation, offline W&B evidence, and newly captured and
scored inputs for all six parity cases. Every case must have correct positions
and maximum absolute response-token log-probability disagreement below
`1.0 nat`. Completion and evidence checks are independent requirements;
editing the source without running training does not pass.

The recorded control maximum is 0.550462 nat (0.449538 below the limit), and
the defective maximum is 3.740287 nat (2.740287 above it). This calibration uses
one run per side. There is no separate validation-reward pass threshold.
Offline replay rejects the recorded defective side and accepts the control;
it does not substitute for the fresh solution evaluation.

## External dataset asset

- Repository: `hiyouga/geometry3k`
- Pinned revision: `fd21e533e1e50d0662a2bf7b223e60511bd5f8b7`
- Destination: `workspace/data/geometry3k_source/`
- Prepared training inputs: `workspace/data/geo3k/`

| Prepared runtime path | Size, bytes | SHA-256 |
| --- | ---: | --- |
| `workspace/data/geo3k/train.parquet` | 42,494,144 | `e769578326ef91738de92181c3d9afe7d0f97f736e3b18e0ecc2158af4e6d9aa` |
| `workspace/data/geo3k/test.parquet` | 12,223,993 | `f4af9e26e23d55499d124e51f7bf70b09685c8848baac3f39ac7981727e63a27` |

`environment/prepare.sh` downloads immutable raw Parquet files, verifies their
SHA-256 and byte sizes, then generates training Parquet and frozen inputs on
CPU. Complete asset checksums are in `environment/asset_pins.json`.

## External model asset

- Repository: `Qwen/Qwen3-VL-8B-Instruct`
- Pinned revision: `fe3340460dc1b2245a90d1b3b608bc86ffb3bfd6`
- Destination: `workspace/models/Qwen3-VL-8B-Instruct/`
- Four model weight shards, approximately 17.5 GB in total

Model files are downloaded and individually checked during prepare. Weights
and reconstructable datasets are not included in the task archive.

## Environment requirement

- Shared image definition: `docker/images/verl-qwen3vl-cu130.Dockerfile`
- Image name: `verl-qwen3vl-cu130`
- Container tag: `autorlbench/verl-qwen3vl-cu130:vllm0.24-torch2.11-cu130`
- Base: digest-pinned `verlai/verl:vllm024.dev2`
- Recorded stack: Python 3.12.13, Torch 2.11 / CUDA 13, vLLM 0.24,
  Transformers 5.5.4, TensorDict 0.10, FlashAttention 2.8.3, Accelerate 1.13.0
- Runtime: 8 H800 GPUs and large shared memory
- Launch script: `workspace/run_qwen3vl_geo3k_grpo.sh`
- Docker validation status: image not yet published; fresh GPU validation pending

GPU reference measurements were obtained in the recorded Python runtime.
CPU asset reconstruction and checksum verification have been tested. The image
is not yet published, and image-based online download plus a fresh GPU
experiment in the declared Docker image remain unverified. The paired experiment took approximately
122 minutes including initialization, parity capture, training, and validation;
each arm spent approximately 39 minutes on training steps alone.

## Reproduce the defective run

Image build installs dependencies. Model and dataset downloads happen in the
networked `environment/prepare.sh` stage, with `https://hf-mirror.com` as the
default Hugging Face endpoint. After prepare, training reads assets inside the
workspace and does not need host-specific model or dataset paths.

From the repository root:

```bash
# Build the declared runtime image.
python3 -m autorlbench build-images verl-qwen3vl-cu130
TASK="$PWD/tasks/infrastructure/qwen3vl_geo3k"
IMAGE=autorlbench/verl-qwen3vl-cu130:vllm0.24-torch2.11-cu130

# Prepare the pinned assets in a networked CPU container. The fetch/dataset
# tools live in the task's environment/ and prepare.sh runs them via
# $AUTORLBENCH_ENVIRONMENT, so that var must point at the mounted dir.
docker run --rm \
  -v "$TASK/workspace:/workspace" \
  -v "$TASK/environment:/autorlbench/environment:ro" \
  -e AUTORLBENCH_WORKSPACE=/workspace \
  -e AUTORLBENCH_ENVIRONMENT=/autorlbench/environment \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 \
  --entrypoint bash "$IMAGE" -lc \
  'bash /autorlbench/environment/prepare.sh'

# Run on eight GPUs and write new experiment logs.
docker run --rm --gpus all --shm-size=16g \
  -v "$TASK/workspace:/workspace" \
  -e LOG_DIR=/workspace/logs/reproduction \
  -e WANDB_DIR=/workspace/logs/reproduction \
  --entrypoint bash "$IMAGE" /workspace/run_qwen3vl_geo3k_grpo.sh
```


If already inside the prepared 8-GPU runtime, the sole training entry is:

```bash
WS=/workspace  # change to the actual workspace directory when running on the host
LOG_DIR="$WS/logs/reproduction" WANDB_DIR="$WS/logs/reproduction" \
  bash "$WS/run_qwen3vl_geo3k_grpo.sh"
```

The launch script runs only the defective arm. It does not download assets or
construct a control workspace. Qualification evidence lives in the hidden
oracle material: the recorded paired-experiment measurements (defective vs
control) are frozen under `evaluations/solution/oracle/recorded_results/` and
`evaluations/issue/verifier/resources/qualification.json`; the hidden
reference source reconstructs a
control workspace using the same launcher and settings.
