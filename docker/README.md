# docker/

Evaluation images and the in-container tooling shared by every task.

| Path | What it is |
| --- | --- |
| `images/<name>.Dockerfile` | One environment per file. A task selects it with `environment.image = "<name>"` in `task.toml`. |
| `autorlbench-activate.sh` | Login-shell hook baked into every image. Restores the image `PATH`, loads `$AUTORLBENCH_PREPARED/env`, and gives the agent a private pip prefix. |
| `install-autorlbench-tooling.sh` | Installs the fetch tooling at prepare time from the repo copy, so fixes take effect without rebuilding images. |
| `autorlbench_fetch.py` | `autorlbench_fetch_hf` / `autorlbench_fetch_url`: pinned-revision downloads with sha256 checks and a shared cache. A mismatch is a hard failure. |
| `autorlbench_dataset.py`, `autorlbench_datasets/` | `autorlbench_dataset`: dataset recipes that need a transform, not just a download. |
| `search-r1-runtime/` | The `uv.lock` the `search-r1-verl` image installs from. |
| `warm-cache.sh` | Runs every task's `prepare.sh` once to fill the model/dataset caches for offline evaluation. |

## Environments

| Name | Base | Used by |
| --- | --- | --- |
| `verl` | `verlai/verl:vllm017.latest` | `data/verl_taco_difficulty_mismatch` |
| `slime-v0.3.2` | `slimerl/slime:v0.3.2-cu129` | `reward/slime_retool_length_reward_bonus` |
| `search-r1-verl` | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` + `search-r1-runtime/uv.lock` | `environment/search_r1_zero_query_embedding` |
| `ppo-critic-vimpo` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | `reward/proxy_reward_misalignment`, `reward/reward_parser_mismatch` |
| `verl` | `verlai/verl:vllm017.latest` | `algorithm/verl_grpo_uid_after_repeat` |
| `verl` | `verlai/verl:vllm017.latest` | `algorithm/verl_grpo_advantage_broadcast` |
| `verl` | `verlai/verl:vllm017.latest` | `configuration/trl_gsm8k_eos_mismatch` |
| `verl` | `verlai/verl:vllm017.latest` | `environment/alfworld_tool_schema_mismatch` |
| `verl` | `verlai/verl:vllm017.latest` | `infrastructure/torch_functional` |
| `verl` | `verlai/verl:vllm017.latest` | `reward/task_function_call` |
| `slime-v0.3.2` | `slimerl/slime:v0.3.2-cu129` | `reward/slime_search_r1_nomask` |
| `verl-moe-token` | `verlai/verl:vllm017.latest` + `verl-moe-token/mcore5047-minimal.patch` | `infrastructure/verl_moe_token_normalization` |
| `verl_checkpoint_host_memory_oom` | `hiyouga/verl` (pinned digest) | `infrastructure/verl_checkpoint_host_memory_oom` |
| `slime-v0.3.2` | `slimerl/slime:v0.3.2-cu129` | `infrastructure/slime_critic_value_head_random_init` |
| `miles-async-rollout` | `radixark/miles` (pinned digest) | `infrastructure/miles_async_rollout_inflight_not_evicted` |
| `trl-grpo-torch26` | `python:3.12.12-slim-bookworm` (pinned digest) | `algorithm/trl_dapo_accumulation_normalization`, `infrastructure/trl_conditional_reward_logging` (not yet built) |
| `justrl2-data-relocation` | `ghcr.io/merak0514/autorlbench-justrl2` (pinned digest) | `data/justrl2_data_relocation` |
| `verl-0.9-vllm0.28` | `vllm/vllm-openai:v0.28.0` (pinned digest) | `reward/verl_grpo_gsm8k_any_number_reward` (not yet built) |
| `verl-0.9-vllm0.28` | `vllm/vllm-openai:v0.28.0` (pinned digest) | `configuration/verl_grpo_ppo_mini_batch_too_small` (not yet built) |
| `verl-vllm011` | `verlai/verl:vllm011.dev5` (pinned digest) | `infrastructure/verl_pissa_vllm_adapter_request` (not yet built) |

An image shared by several tasks is part of each task's ground truth. If you change a
Dockerfile, rerun every task that uses it. A task that needs a different stack gets a new
environment name.

## Building

```bash
python3 -m autorlbench build-images              # build missing images
python3 -m autorlbench build-images --rebuild verl
```

The tag comes from the `# autorlbench-image:` header of each Dockerfile and is not derived
from its content, so pass `--rebuild` after you edit one. The build context is `docker/`,
restricted by `.dockerignore` to an allowlist.

Rules for a new environment:

1. Pin `FROM` by digest (`image:tag@sha256:...`).
2. Copy the block after `# ---- AutoRLBench tooling: keep identical in every docker/images/*.Dockerfile ----`
   verbatim. A test checks that this block is identical across files.

## Stages

| Stage | Runs | Network | `/autorlbench/{models,datasets}` |
| --- | --- | --- | --- |
| `prepare` | the task's `environment/prepare.sh` | yes | read-write |
| `agent` | the agent in `/workspace` | per backend policy | read-only |
| `verifier` | the selected `verifier/test.sh` | per backend policy | read-only |

All three stages use the same image, each in a separate container. Hidden oracle and verifier
files are mounted only into the verifier container.
