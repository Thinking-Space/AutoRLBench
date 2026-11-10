#!/usr/bin/env bash
# Qwen1.5-MoE-A2.7B-Chat, decoupled PPO with IcePop token masking on GSM8K (AReaL, local scheduler).
# 3 epochs = 87 steps on one 8-GPU node: 4 SGLang rollout servers + 4 FSDP actor ranks
# (the allocation and every other default come from examples/math/gsm8k_icepop.yaml).
# The recorded run resolved the model and dataset by hub id from an offline cache
# (actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat, dataset path openai/gsm8k, see config.yaml);
# here they point at the local copies that environment/prepare.sh places in the workspace.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$W"
export PYTHONPATH="$W${PYTHONPATH:+:$PYTHONPATH}"

SEED="${SEED:-2}"
RLROOT="${RLROOT:-$W/run}"
MODEL_PATH="${MODEL_PATH:-$W/models/Qwen1.5-MoE-A2.7B-Chat}"
DATA_PATH="${DATA_PATH:-$W/data/gsm8k}"
mkdir -p "$RLROOT/experiments" "$RLROOT/name_resolve" "$RLROOT/logs"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-token logprob dumps (inference-engine / proximal / current), every 20th call.
export KT_DUMP_DIR="$RLROOT/ktdump/seeds_icepop_w067_s${SEED}"
export KT_DUMP_EVERY=20
# Only read by disk-mode weight updates; this run syncs weights over NCCL (xccl).
export KT_WEIGHT_UPDATE_DIR="/dev/shm/${USER:-root}-wu-$$"

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-seeds \
    trial_name=icepop_w067-s${SEED} \
    seed=$SEED \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path="$MODEL_PATH" \
    train_dataset.path="$DATA_PATH" \
    valid_dataset.path="$DATA_PATH" \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    ++rollout.setup_timeout=900 \
    total_train_epochs=3 \
    actor.weight_update_mode=xccl \
    actor.request_timeout=3600 \
    rollout.request_timeout=3600 \
    vllm.disable_custom_all_reduce=false \
    actor.rejection_sampling.lower=0.67 \
    actor.rejection_sampling.upper=1.5
