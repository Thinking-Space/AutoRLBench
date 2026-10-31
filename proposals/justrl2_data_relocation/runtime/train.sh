#!/usr/bin/env bash
set -euo pipefail
: "${RELOCATION_CODE:?}" "${RELOCATION_DATA:?}" "${RELOCATION_OUTPUT:?}" "${RELOCATION_MODEL:?}" "${RELOCATION_NATIVE:?}"
cd "$RELOCATION_CODE"
source justrl2/model_args/minicpm5-2b.sh
export PYTHONPATH="$RELOCATION_PLUGIN:$PWD:$PWD/Megatron-LM:$PWD/sglang/python"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0 NVTE_UNFUSED_ATTN=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0 RAY_DEDUP_LOGS=0 WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export BENCH_EVIDENCE_DIR="$RELOCATION_OUTPUT/optimizer_evidence"
export TENSORBOARD_DIR="$RELOCATION_OUTPUT/tensorboard"
STEPS=${RELOCATION_STEPS:-8}
BATCH=${RELOCATION_BATCH:-8}
SAMPLES=${RELOCATION_SAMPLES:-2}
MAX_RESPONSE=${RELOCATION_MAX_RESPONSE:-8192}
CONTEXT=$((MAX_RESPONSE + 2048))
mkdir -p "$RELOCATION_OUTPUT"
python3 "$RELOCATION_PLUGIN/run_native.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
  --critic-num-nodes 1 --critic-num-gpus-per-node 1 --num-gpus-per-node 3 \
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 \
  --train-backend megatron --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --qkv-format thd --attention-backend fused --no-gradient-accumulation-fusion \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$RELOCATION_MODEL" --ref-load "$RELOCATION_NATIVE" \
  --load "$RELOCATION_OUTPUT/actor" --save "$RELOCATION_OUTPUT/actor" \
  --critic-save "$RELOCATION_OUTPUT/critic" --save-interval "$RELOCATION_AT" \
  --save-retain-interval 100000 \
  --prompt-data "$RELOCATION_DATA/eval.jsonl" --input-key prompt --label-key label \
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":true}' \
  --data-source-path relocation_rollout.BackedMathSource \
  --rollout-function-path relocation_rollout.generate_rollout \
  --custom-generate-function-path relocation_rollout.generate \
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted \
  --rm-type math --num-rollout "$STEPS" --start-rollout-id 0 \
  --rollout-batch-size "$BATCH" --over-sampling-batch-size "$BATCH" \
  --n-samples-per-prompt "$SAMPLES" --global-batch-size "$((BATCH * SAMPLES))" --micro-batch-size 1 \
  --rollout-max-prompt-len 2048 --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-max-context-len "$CONTEXT" --seq-length "$CONTEXT" \
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-stop-token-ids 1 130073 \
  --advantage-estimator ppo --normalize-advantages --calculate-per-token-loss \
  --gamma 1 --gae-lambda-k 0.5 --critic-value-bias-init 0.52 \
  --num-critic-only-steps 0 --critic-lr-warmup-iters 0 \
  --optimizer adam --lr 1e-6 --critic-lr 5e-6 --lr-decay-style constant \
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --eps-clip 0.2 --eps-clip-high 0.28 --value-clip 0.2 \
  --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --attention-dropout 0 --hidden-dropout 0 --log-probs-chunk-size 1024 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --sglang-mem-fraction-static 0.65 --sglang-max-running-requests 32 \
  --sglang-max-total-tokens "${RELOCATION_MAX_TOTAL_TOKENS:-262144}" --sglang-context-length "$CONTEXT" \
  --sglang-attention-backend flashinfer --sglang-log-level warning \
  --update-weights-interval 1 --distributed-timeout-minutes 10 \
  --eval-prompt-data fixed "$RELOCATION_DATA/eval.jsonl" --eval-interval "$RELOCATION_AT" \
  --n-samples-per-eval-prompt 1 --eval-temperature 0.0 --eval-max-response-len "$MAX_RESPONSE" \
  --seed 20261031 --rollout-seed 20261031 --use-tensorboard \
  --save-debug-rollout-data "$RELOCATION_OUTPUT/rollout_{rollout_id}.pt"
