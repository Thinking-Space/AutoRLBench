#!/bin/bash
#SBATCH --job-name=teacher-rollout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=/historical/shared/qat/r/rollout/rollout-%j.out
#SBATCH --error=/historical/shared/qat/r/rollout/rollout-%j.err
set -euo pipefail

# Teacher trajectory rollout for the off-policy QAT arm.
#
# Each shard is its own single-GPU vLLM engine (TP=1).  TP>1 would only add
# collective overhead here: the 1.5B fits on one L20D with room for the KV
# cache of 64 concurrent 16k sequences, and the shards are embarrassingly
# parallel.  CPU/memory are deliberately small so the job can share a node
# with a running training job -- asking for 128 CPUs / 900G would make it
# wait for a whole free node.
#
# Because the node is shared, the shard loop must index into SLURM's
# allocation mask rather than assume GPUs 0..N-1.  Job 8336 set
# CUDA_VISIBLE_DEVICES=0 on a node where SLURM had handed out GPUs 4-7 and
# GPU 0 held another job's 234 GiB; vLLM died with "Free memory on device
# cuda:0 (32.91/267.69 GiB) ... less than desired GPU memory utilization".

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/vendor/verl:/historical/shared/qat/tmp/turboquant-train-overlay:/historical/shared/qat/envs/mopd-pylibs"
# Triton JIT-compiles cuda_utils.c at engine init and the nodes ship no
# python3-devel headers.
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC" C_INCLUDE_PATH="$PYINC" CPLUS_INCLUDE_PATH="$PYINC"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" /historical/shared/qat/r/rollout

MODEL="${MODEL:-/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl}"
DATA="${DATA:-/historical/shared/qat/data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet}"
OUT_DIR="${OUT_DIR:-/historical/shared/qat/data/rollouts/teacher-1p5b-9600}"
COUNT="${COUNT:-9600}"
SHARDS="${SHARDS:-8}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-42}"
PROMPT_SEED="${PROMPT_SEED:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

mkdir -p "$OUT_DIR"
echo "[rollout] model=$MODEL count=$COUNT shards=$SHARDS out=$OUT_DIR"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# SLURM's mask, e.g. "4,5,6,7" on a shared node.  Never overwrite it with
# absolute indices.
IFS=',' read -r -a ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
echo "[rollout] SLURM allocated GPUs: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if [[ ${#ALLOC_GPUS[@]} -lt $SHARDS ]]; then
  echo "[rollout] SHARDS=$SHARDS but only ${#ALLOC_GPUS[@]} GPU(s) allocated" >&2
  exit 2
fi

pids=()
for ((i = 0; i < SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES="${ALLOC_GPUS[$i]}" python "$ROOT/scripts/rollout_teacher_traj.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --out "$OUT_DIR/shard-$i.parquet" \
    --count "$COUNT" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --prompt-seed "$PROMPT_SEED" \
    --shard-index "$i" \
    --shard-count "$SHARDS" \
    --tensor-parallel-size 1 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    > "$OUT_DIR/shard-$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
if [[ $fail -ne 0 ]]; then
  echo "[rollout] at least one shard failed; NOT merging" >&2
  tail -n 25 "$OUT_DIR"/shard-*.log >&2
  exit 1
fi

python - "$OUT_DIR" "$COUNT" <<'PY'
import glob
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

out_dir, expected = Path(sys.argv[1]), int(sys.argv[2])
shards = sorted(glob.glob(str(out_dir / "shard-*.parquet")))
tables = [pq.read_table(s) for s in shards]
merged = pa.concat_tables(tables)

# A silent short merge would look exactly like a successful run and would
# quietly give the offline arm fewer sequences per step than the OPD arm.
if merged.num_rows != expected:
    raise SystemExit(
        f"merged {merged.num_rows} trajectories from {len(shards)} shards, "
        f"expected {expected}; refusing to write a short dataset"
    )

pq.write_table(merged, out_dir / "trajectories.parquet")
lens = merged.column("response_tokens").to_pylist()
trunc = sum(merged.column("truncated").to_pylist())
summary = {
    "trajectories": merged.num_rows,
    "shards": len(shards),
    "mean_response_tokens": sum(lens) / len(lens),
    "max_response_tokens": max(lens),
    "truncated": trunc,
    "truncated_fraction": trunc / merged.num_rows,
    "total_response_tokens": sum(lens),
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print("[rollout]", json.dumps(summary))
PY

echo "[rollout] done -> $OUT_DIR/trajectories.parquet"
