#!/bin/bash
# GSM8K eval (before) -> GRPO -> GSM8K eval (after). Paths relative to this workspace.
set -uo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-models/Qwen2.5-1.5B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/grpo}
export LOG_DIR=${LOG_DIR:-results}
MAX_STEPS=${MAX_STEPS:-100}; LR=${LEARNING_RATE:-1e-5}; SEED=${SEED:-20261120}; SKIP_EVAL=${SKIP_EVAL:-0}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_DIR"
if [ "$SKIP_EVAL" != 1 ]; then
  python eval_gsm8k.py --model "$MODEL" --step 0 --out "$LOG_DIR/eval_before.json" > "$LOG_DIR/eval_before.log" 2>&1 || { tail -30 "$LOG_DIR/eval_before.log"; exit 1; }
  cat "$LOG_DIR/eval_before.json"; echo; grep "^eval 0:" "$LOG_DIR/eval_before.log"
fi
python train_grpo.py --model "$MODEL" --output "$OUTPUT_DIR" --max-steps "$MAX_STEPS" --learning-rate "$LR" --seed "$SEED" > "$LOG_DIR/train.log" 2>&1
rc=$?; echo "train_grpo.py exit=$rc"; [ $rc -eq 0 ] || { tail -40 "$LOG_DIR/train.log"; exit $rc; }
if [ "$SKIP_EVAL" != 1 ]; then
  python eval_gsm8k.py --model "$OUTPUT_DIR" --step "$MAX_STEPS" --out "$LOG_DIR/eval_after.json" > "$LOG_DIR/eval_after.log" 2>&1 || { tail -30 "$LOG_DIR/eval_after.log"; exit 1; }
  cat "$LOG_DIR/eval_after.json"; echo; grep "^eval $MAX_STEPS:" "$LOG_DIR/eval_after.log"
fi
