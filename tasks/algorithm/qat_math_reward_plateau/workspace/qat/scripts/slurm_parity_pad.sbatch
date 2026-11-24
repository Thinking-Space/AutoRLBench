#!/bin/bash
#SBATCH --job-name=qat-pad
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/historical/shared/qat/r/qat-parity/pad-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-parity/pad-%j.err
#
# Is the actor/rollout log-prob bias caused by veRL's left-padded batch layout?
# Reuses an existing vLLM run's tokens, so only the HF phase runs.
set -euo pipefail
unset LD_PRELOAD
ROOT=/historical/home/QAT
VENV_BIN=/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin
export PATH="$VENV_BIN:$PATH" PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
INCLUDE=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$INCLUDE" C_INCLUDE_PATH="$INCLUDE" CPLUS_INCLUDE_PATH="$INCLUDE"
cd "$ROOT"
SRC="${SRC:?set SRC to a directory containing tq_vllm.json}"
# veRL pads every prompt to data.max_prompt_length = 2048.
PAD=$(( 2048 - $(python3 -c "import json;print(len(json.load(open('$SRC/tq_vllm.json'))['prompt_token_ids']))") ))
echo "[pad] left-padding by $PAD tokens (prompt region -> 2048)"
$VENV_BIN/python scripts/parity_forward_logprobs.py --phase hf \
  --tokens "$SRC/tq_vllm.json" --pad-left "$PAD" --out "$SRC/tq_hf_padded.json"
echo "[pad] done: $SRC/tq_hf_padded.json"
