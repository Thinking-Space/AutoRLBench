<div align="center">

<img src="https://raw.githubusercontent.com/radixark/miles/main/imgs/miles_logo.png" alt="Miles Logo" width="550">

### **Enterprise-Grade Reinforcement Learning for Large-Scale Model Training**
### **High-Performance Rollout • Low Precision Training • Production Stability**

[![GitHub Repo](https://img.shields.io/badge/github-radixark%2Fmiles-black?logo=github)](https://github.com/radixark/miles)
[![License](https://img.shields.io/github/license/radixark/miles)](LICENSE)
[![Slack](https://img.shields.io/badge/slack-join-brightgreen.svg)](https://slack.sglang.ai)

[**Latest Updates**](#latest-updates) | [**Quick Start**](#quick-start) | [**Key Features**](#key-features) | [**Documentation**](https://www.radixark.com/miles/docs)

</div>

---


## Latest Updates

*   **[2026/02]** 💡 **Miles Detailed Arguments**: We've added a detailed command-line argument guide used to configure Miles for RL training and inference. These arguments enable precise control over cluster resources, training backends (Megatron/FSDP), inference optimization via SGLang, and RL algorithmic hyperparameters. [Link](https://github.com/radixark/miles/blob/main/docs/en/advanced/miles_server_args.md)
*   **[2026/01]** 💎 **INT4 Quantization-Aware Training (QAT)**: Inspired by the Kimi K2-Thinking report, Miles now features a full-stack INT4 W4A16 QAT pipeline. This allows 1TB-scale models to fit into single-machine VRAM (e.g., NVIDIA H200), doubling rollout efficiency by eliminating cross-node bottlenecks while maintaining BF16-equivalent accuracy. [Blog](https://lmsys.org/blog/2026-01-26-int4-qat/)
*   **[2026/01]** 💎 **Unified VLM/LLM Multi-Turn Training**: We provided an implementation for the VLM multi-turn sampling paradigm. Developers only need to write a customized `rollout` function to easily start multi-turn RL for VLM, just like training LLM. [Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/vlm-multi-turn/readme-en.md)
*   **[2026/01]** 🤖 **Multi-Agent Co-Evolution**: Miles now supports **MrlX**, a novel asynchronous co-evolutionary framework for Multi-Agent RL. Achieve superior performance in complex tasks like Doctor-Patient simulations and DeepResearch pipelines by enabling specialized agents to evolve together symbiotically. [[Link]](https://github.com/AQ-MedAI/MrlX)
*   **[2025/12]** 🔄 **Rollout Routing Replay (R3)**: In collaboration with SGLang, we've launched R3 to solve MoE RL instability. R3 records inference routing decisions and replays them during training, effectively eliminating the "training-inference mismatch" and preventing training collapse in large MoE models like Qwen3 and DeepSeek-V3. [[Paper]](https://arxiv.org/pdf/2510.11370) [[Docs]](docs/en/advanced/miles-router.md#22-rollout-routing-replay-r3-for-moe)
*   **[2025/11]** 🔥 **Unified FP8 Release**: Solves the stability issues in MoE RL by ensuring training and inference use the exact same FP8 quantization logic. [[Blog]](https://lmsys.org/blog/2025-11-25-fp8-rl/)
*   **[2025/11]** ⚡ **Speculative Decoding in RL**: Integrated speculative rollout with online SFT for draft models, achieving massive throughput gains. [[Blog]](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/spec/readme-en.md)
*   **[2025/11]** 🎉 **Miles Project Launch**: A joint effort by InfiXAI, Ant Group, SGLang RL Team, and the Miles community. [[Announcement]](https://lmsys.org/blog/2025-11-19-miles/)

## What is Miles?

**Miles** is a high-performance, enterprise-ready reinforcement learning (RL) framework specifically optimized for **Large-Scale model Post-Training**. Built as a powerful fork of **[slime](https://github.com/THUDM/slime)**, Miles bridges the gap between research-grade RL and production-grade reliability by integrating **SGLang** for high-throughput rollout and **Megatron-LM** for scalable training.

> *"A journey of a thousand miles begins with a single rollout."* — Miles focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

---


## Key Features

### 🌪️ Advanced MoE & Low-Precision Training

*   **Unified FP8 Pipeline**: The first framework to implement end-to-end FP8 sampling and training. By unifying precision across rollout and training, Miles eliminates the quantization-induced discrepancy that causes RL collapse in large MoE models.
*   **Rollout Routing Replay (R3)**: Records expert routing decisions during SGLang inference and replays them during training to ensure bit-wise expert alignment.
*   **INT4 QAT Support**: Recommendation for 1TB+ models to enable single-machine (e.g., H200) deployment by significantly reducing memory footprint.

### 🛡️ Eliminating Train-Inference Mismatch

*   **Bit-wise Identical Training and Inference Log Probs**: System-level solution achieving deterministic forward/backward passes through kernel-level optimization (FlashAttention-3, DeepGEMM).
*   **Algorithmic Correction (TIS/MIS)**: When mismatch is unavoidable, Miles provides **Truncated Importance Sampling (TIS)** and **Masked Importance Sampling (MIS)** to mitigate off-policy bias and prevent training divergence.

### ⚡ Extreme Performance & Efficiency

*   **Speculative RL Training**: Achieve **25%+ rollout speedup** by using an **Online SFT Draft Model**. Unlike frozen draft models, Miles updates the draft policy during RL to prevent policy drift.
*   **Zero-Copy Weight Sync**: Optimized weight refit via **CUDA IPC zero-copy mapping**, async tensor gathering, and bucketed flattening. Sync time reduced by 50% compared to standard HTTP/RPC transfers.
*   **Partial Rollout & Over-Sampling**: Handles the "Long-Tail Effect" in multi-turn RL by over-sampling requests and recycling half-finished trajectories to maximize GPU utilization.

## Model Support & Training Diversity

### 🏗️ Supported Models
Miles supports a wide range of state-of-the-art architectures, with a special emphasis on **DeepSeek, Qwen, Llama** and mainstream models.

| Family | Supported Models |
| :--- | :--- |
| **DeepSeek** | **R1, V3, V3.2** |
| **Qwen** | **Qwen 2, 2.5, 3** |
| **Llama** | **Llama 3, 3.1, 3.3, 4** |
| **Gemma** | **Gemma 2, 3, 3N** |
| **GLM** | **GLM-4.5, GLM-4.6, GLM-4.7** |
| **MiniMax** | **M2, M2.1** |
| **Others** | **Mistral, Mixtral, Phi, gpt-oss and any model supported by SGLang and Megatron** |

### 🧩 Diverse Training Scenarios
Miles is designed to handle the complexity of modern RL workloads across various dimensions:
*   **Multi-Turn Interaction**: Optimized for complex, multi-round conversations and tool-use scenarios.
*   **VLM & LLM Support**: Unified framework for both Vision-Language and pure Text models.
*   **Reasoning & Coding**: Specific recipes and optimizations for **Reasoning (Math/Logic)** and **Coding Agent** tasks.
*   **Multi-Agent Training**: Support for advanced co-training and collaborative multi-agent reinforcement learning.

---

## Quick Start

### Installation

We recommend using our official Docker image for the best performance and compatibility:

```bash
# Pull the latest image
docker pull radixark/miles:latest

# Or install from source
pip install -r requirements.txt
pip install -e .
```

### Launch Training

Miles provides a unified entry point for complex RL tasks. Here is an example of FP8 GRPO training for Qwen3:

```bash
python train.py \
    --advantage-estimator grpo \
    --model-name qwen3-30b-a3b \
    --hf-checkpoint /path/to/qwen3-30b-a3b-hf \
    --rollout-batch-size 512 \
    --n-samples-per-prompt 8
```

For comprehensive guides on environment setup and custom reward functions, see the [Quick Start Guide](docs/en/get_started/quick_start.md).

---

## Roadmap

### ✅ Completed

- [x] **Unified FP8** E2E Training & Rollout
- [x] **INT4 Quantization-Aware Training (QAT)**: Single-machine 1TB models
- [x] **Speculative RL** with Online SFT
- [x] **Multi-Agent RL** (Co-evolutionary frameworks like [MrlX](https://github.com/AQ-MedAI/MrlX))
- [x] **Support DeepSeek V3.2 Models**
- [x] **VLM Multi-Turn Training**
- [x] **Aligning SGLang with Megatron in Dense Models**
- [x] **Rollout Routing Replay (R3)**

### 🏗️ In Progress & Planned

- [ ] **Zero mismatch for MoE RL**
- [ ] **Aligning SGLang with Megatron in MoE Models**
- [ ] **Diffusion RL** Support
- [ ] **Omni RL** Support
- [ ] **Diffusion LLM RL** Support
- [ ] **Elastic Resource Scheduling**: Dynamic scaling of rollout vs. training workers



---

## Local Experiment Notes: Qwen3-8B Kernel Bench (2026-04)

The following notes summarize the local ablation runs performed in this workspace for the Qwen3-8B kernel-bench RL setup. They are operational notes for reproducibility and debugging, not upstream benchmark claims.

### Main Findings

- Continuous reward produced materially stronger non-zero policy signals than binary reward. `244147` and `245774` both avoided the classic binary `advantages ~= 0` failure mode.
- Binary baseline runs (`244258`, `244625`) often preserved non-zero `raw_reward`, but group-normalized `rewards`, `advantages`, and `pg_loss` still collapsed close to zero. In practice, the policy was barely updating.
- The signal-enhanced binary recipe (`245772`, `245773`) improved gradient availability. Mixed-group sampling worked, `solve_none` / `solve_all` dead zones were reduced, and `grpo_metrics/no_grad` stayed healthy in the successful early steps.
- FSDP continuous remained memory-bound. `244364` and `245774` both failed in actor backward even after progressively more conservative memory settings.
- Sandbox / reward-pipeline instability was a recurring bottleneck. Several runs spent large fractions of wall-clock time blocked by `Sandbox unreachable: ZMQ timeout`.
- The resume experiment `246033` did not reach the first rollout. It failed during Megatron actor initialization because the checkpoint learning rate (`5e-6`) did not match the resumed script's configured learning rate (`2e-6`).

### Experiment Matrix

| Task | Script / configuration | Status | Key changes | Main observation |
| :--- | :--- | :--- | :--- | :--- |
| `244147` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-continuous.sh` | `FAILED` | Megatron + continuous reward baseline, `n_samples=8`, `temp=0.8`, `entropy=0.0` | Healthiest training signal among the early runs; progressed to roughly `step 75`; failed on rollout-to-SGLang HTTP connectivity rather than training collapse. |
| `244176` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-binary.sh` | `FAILED` | First FSDP binary port before signal-enhanced changes | Failed in FSDP actor backward with CUDA OOM. |
| `244177` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-continuous.sh` | `FAILED` | First FSDP continuous port | Training ran, then failed during distributed checkpoint save with `NCCL Error 1`. |
| `244258` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench.sh` | `STOPPED` | Megatron + binary baseline | Useful as a control only: non-zero `raw_reward`, but `advantages` and `pg_loss` were near zero, so the policy was barely learning. |
| `244364` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-continuous.sh` | `FAILED` | FSDP continuous retry with periodic save removed | Avoided the checkpoint-save failure from `244177`, but still failed in actor backward due to OOM. |
| `244625` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-binary.sh` | `STOPPED` | FSDP binary OOM-fix (`max_tokens=8192`, `sglang_mem=0.40`, `max_response_len=12288`) | No early backward OOM, but learning signal was still close to zero after reward normalization. |
| `245772` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-binary.sh` | `STOPPED` | Signal-enhanced FSDP binary: `n_samples=16`, `temp=0.6`, `entropy=0.005`, easy injection, mixed-group retry, no-grad early stop | Best binary line so far: dead-zone metrics improved substantially, but policy updates were still weak after normalization. |
| `245773` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench.sh` | `STOPPED` | Signal-enhanced Megatron binary using the same recipe as `245772` | Reached at least `step 6`, then spent a long time blocked in rollout / reward pipeline; no formal checkpoint was saved. |
| `245774` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-fsdp-continuous.sh` | `FAILED` | More aggressive FSDP continuous memory cut (`max_tokens=6144`, `sglang_mem=0.35`, `max_response_len=8192`) | Continuous reward produced real non-zero signal up to `step 43`, but `no_grad` remained high and actor backward OOMed at `rollout 44`. |
| `246033` | `scripts/community/run-qwen3-8b-16gpu-kernel-bench-resume-239883-iter99.sh` | `FAILED` | Resume from source run `239883` at `iter_0000099`, with pre-signal-enhancement binary settings restored | Failed before the first rollout because the resumed checkpoint expected `lr=5e-6` while the script configured `lr=2e-6`. |

### Signal-Enhanced Binary Recipe

Compared with the original binary baseline, the signal-enhanced line introduced the following changes:

- `n_samples_per_prompt: 8 -> 16`
- `rollout_temperature: 0.8 -> 0.6`
- `entropy_coef: 0.00 -> 0.005`
- add `kernel_bench_easy_injection.jsonl` to the training mix
- use `check_reward_mixed_group` instead of `check_reward_nonzero_std`
- retry the same prompt up to `3` times if a group is `solve_none` or `solve_all`
- enable early stop when `grpo_metrics/no_grad > 0.8` for `3` consecutive rollouts

This recipe improved gradient availability in binary reward runs, but did not yet fully solve the weak-update problem after reward normalization.

### Decision Table

| Line | Recommendation | Reason |
| :--- | :--- | :--- |
| Megatron + continuous (`244147`) | Keep as the main signal-quality reference | Training signal was healthy; failure was infrastructure-side rather than optimization-side. |
| FSDP + binary + signal-enhanced (`245772`) | Keep as the main binary line | Best evidence so far that binary reward can be made trainable with mixed-group and easy-sample tricks. |
| FSDP + continuous (`245774`) | Pause and reduce memory peak further before retrying | Signal quality is promising, but current memory headroom is still insufficient. |
| Megatron / FSDP binary baselines (`244258`, `244625`) | Archive as controls; do not continue spending compute | They establish the baseline failure mode but are not competitive training recipes. |
| Resume from `239883@99` (`246033`) | Fix optimizer / scheduler resume mismatch before retrying | Current run never reached training due to a checkpoint-vs-config learning-rate assertion. |

### Important Artifact Paths

- source checkpoint for resume experiments: `checkpoints/qwen3_8b_kernel_bench_16gpu_20260408_095936`
- Megatron continuous baseline: `checkpoints/qwen3_8b_kernel_bench_continuous_16gpu_20260410_050347`
- Megatron binary baseline: `checkpoints/qwen3_8b_kernel_bench_16gpu_20260410_163258`
- signal-enhanced Megatron binary: `checkpoints/qwen3_8b_kernel_bench_16gpu_20260411_095522`
- aggressive-memory FSDP continuous: `checkpoints/qwen3_8b_kernel_bench_fsdp_continuous_16gpu_20260411_124910`
- resume attempt from `239883@99`: `checkpoints/qwen3_8b_kernel_bench_resume239883_iter0099_20260412_081925`

## Acknowledgements

Miles is built upon the shoulders of giants in the LLM infrastructure ecosystem:
*   **[slime](https://github.com/THUDM/slime)**: The core modular architecture and inspiration.
*   **[SGLang](https://github.com/sgl-project/sglang)**: The high-performance inference engine.
*   **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: Robust large-scale training components.

Special thanks to **InfiXAI Team**, **Ant Group AQ Team**, **SGLang RL Team**, and the **Miles Team**. We also thank **DataCrunch** for compute sponsorship and **NVIDIA** for technical support on Transformer Engine (TE).

---

## Links

*   **GitHub**: [https://github.com/radixark/miles](https://github.com/radixark/miles)
*   **Slime Project**: [https://github.com/THUDM/slime](https://github.com/THUDM/slime)
*   **Developer Guide**: Check the `docs/` and `examples/` directories for in-depth technical notes.

<div align="center">

**Give Miles a ⭐️ Star if it helps your RL journey!**

</div>
