#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inside the prepared JustRL2 workspace. Values are shared by both arms.
set -euo pipefail
: "${BENCH_MODEL:?point to the pinned public HF model}"
: "${BENCH_NATIVE_MODEL:?point to its converted torch_dist root}"
: "${BENCH_REPLAY:?point to the frozen replay.pt}"
: "${BENCH_RUN_DIR:?use a fresh output directory}"
cd "$(dirname "$0")/.."
mkdir -p "$BENCH_RUN_DIR"
export BENCH_EVIDENCE_DIR="$BENCH_RUN_DIR/evidence"
export PYTHONPATH="$PWD:$PWD/Megatron-LM:$PWD/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0
export NVTE_UNFUSED_ATTN=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$BENCH_RUN_DIR/nccl_trace_rank_"
export WANDB_MODE=offline
export RAY_DEDUP_LOGS=0
source justrl2/model_args/minicpm5-2b.sh

LOAD_ARGS=(--load "$BENCH_RUN_DIR/actor")
ROLLOUT_ARGS=(--num-rollout 1 --start-rollout-id 0)
if [[ -n "${BENCH_RESTORE_ROOT:-}" ]]; then
  : "${BENCH_RESTORE_ITERATION:?restore requires the original iteration}"
  [[ "$BENCH_RESTORE_ITERATION" =~ ^[0-9]+$ ]]
  LOAD_ARGS=(--load "$BENCH_RESTORE_ROOT/actor" --critic-load "$BENCH_RESTORE_ROOT/critic"
             --use-checkpoint-opt_param-scheduler)
  ROLLOUT_ARGS=(--num-rollout "$((BENCH_RESTORE_ITERATION + 2))")
fi

# Native replay bypasses SGLang generation but executes the real actor/critic
# forward passes, value exchange, JustRL2 GAE, backward, Adam and checkpoint save.
python3 benchmark/run_training.py -- \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 \
  --critic-num-nodes 1 --critic-num-gpus-per-node 4 --num-gpus-per-node 8 \
  --train-backend megatron --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 --context-parallel-size 2 \
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --qkv-format thd --attention-backend fused \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$BENCH_MODEL" --ref-load "$BENCH_NATIVE_MODEL" \
  "${LOAD_ARGS[@]}" --save "$BENCH_RUN_DIR/actor" \
  --critic-save "$BENCH_RUN_DIR/critic" \
  --save-interval 1 --save-retain-interval 100000 \
  --load-debug-rollout-data "$BENCH_REPLAY" --disable-rollout-global-dataset \
  --rollout-function-path benchmark.replay_loader.unused_rollout \
  --eval-function-path benchmark.replay_loader.unused_rollout \
  "${ROLLOUT_ARGS[@]}" --rollout-batch-size 2 \
  --n-samples-per-prompt 1 --global-batch-size 2 --micro-batch-size 1 \
  --rollout-max-prompt-len 2048 --rollout-max-response-len 2048 \
  --rollout-max-context-len 4096 --seq-length 4096 \
  --advantage-estimator ppo --normalize-advantages --calculate-per-token-loss \
  --gamma 1 --gae-lambda-k 0.5 --critic-value-bias-init 0.52 \
  --num-critic-only-steps 0 --critic-lr-warmup-iters 0 \
  --optimizer adam --lr 1e-6 --critic-lr 5e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --eps-clip 0.2 --eps-clip-high 0.28 --value-clip 0.2 \
  --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --attention-dropout 0 --hidden-dropout 0 --log-probs-chunk-size -1 \
  --distributed-timeout-minutes 1 --seed 20261028 \
  --save-debug-rollout-data "$BENCH_RUN_DIR/rollout_{rollout_id}.pt" \
  --save-debug-train-data "$BENCH_RUN_DIR/train_{rollout_id}_{rank}.pt"
