#!/bin/bash
#SBATCH --job-name=nvfp4-ptq-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l20d:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/historical/shared/qat/r/nvfp4/ptq-eval-%j.out
#SBATCH --error=/historical/shared/qat/r/nvfp4/ptq-eval-%j.err
set -uo pipefail

# NVFP4 PTQ + AIME 2024 avg@32, four tiers, one 8-GPU node.
#
#   bf16        base model,           kv auto
#   w4a16kv16   NVFP4 weights,        kv auto
#   w4a16kv4    NVFP4 weights,        kv nvfp4
#   w4a4kv4     NVFP4 weights+acts,   kv nvfp4
#
# Two PTQ checkpoints cover four eval configs: the KV tier is a vLLM serving
# flag, not a property of the checkpoint.
#
# tensor_parallel_size is pinned to 1 for every config.  On this benchmark TP is
# not a free knob: the same 30 problems at the same seed moved 1.67 pp between
# TP=2 and TP=1, because the reduction order changes the rounding and therefore
# the sampled tokens.  Comparing tiers evaluated at different TP would fold that
# into the result.
#
# Shard loop indexes into SLURM's CUDA_VISIBLE_DEVICES rather than assuming
# 0..N-1; overriding it with absolute indices once pointed a job at another
# job's GPU (job 8336).

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$ROOT:/historical/shared/qat/envs/modelopt-pylibs:/historical/shared/qat/envs/mopd-pylibs:/historical/shared/qat/tmp/turboquant-train-overlay"
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC" C_INCLUDE_PATH="$PYINC" CPLUS_INCLUDE_PATH="$PYINC"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/historical/shared/qat/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/historical/shared/qat/cache/triton}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" /historical/shared/qat/r/nvfp4

MODEL="${MODEL:-/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl}"
CKPT_ROOT="${CKPT_ROOT:-/historical/shared/qat/models/nvfp4-1p5b}"
OUT_DIR="${OUT_DIR:-/historical/shared/qat/runs/qat-eval-1p5b-nvfp4-aime24-avg32}"
CALIB="${CALIB:-/historical/shared/qat/data/rollouts/teacher-1p5b-9600/trajectories.parquet}"
CALIB_SAMPLES="${CALIB_SAMPLES:-1000}"
DATA=/historical/shared/qat/data/raw/aime_2024/data/train-00000-of-00001.parquet
N="${N:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-33792}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

[[ -f "$DATA" ]] || { echo "missing dataset: $DATA" >&2; exit 2; }
[[ -f "$CALIB" ]] || { echo "missing calibration data: $CALIB" >&2; exit 2; }
mkdir -p "$CKPT_ROOT" "$OUT_DIR/eval"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
echo "[nvfp4] SLURM GPUs: ${CUDA_VISIBLE_DEVICES:-<unset>}  (${#GPUS[@]} visible)"
nvidia-smi --query-gpu=index,name,compute_cap,memory.used --format=csv,noheader | head -8

# Check only the GPUs SLURM handed us, not the whole node: this job may hold a
# subset, and a neighbour's memory on an index we were not given is none of our
# business.  Verify rather than assume even with --exclusive -- lj-gpu5 sat
# "idle" in sinfo with four orphaned VLLM::EngineCore processes holding 246 GiB
# each, invisible to the scheduler, which is what killed job 8796.
# Keep only the GPUs that are actually free.  This partition oversubscribes,
# so a strict "all allocated GPUs must be idle" check turns into repeated
# one-second failures and hand resubmission; job 8941 died that way.  Ask for
# more GPUs than tiers and drop the busy ones instead.
FREE=()
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
  if [[ -n "$used" && "$used" -le 4096 ]]; then FREE+=("$g"); else
    echo "[nvfp4] skipping gpu $g (${used}MiB in use by another job)"
  fi
done
GPUS=("${FREE[@]}")
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[nvfp4] no free GPU among ${CUDA_VISIBLE_DEVICES}" >&2
  exit 3
fi
echo "[nvfp4] usable GPUs: ${GPUS[*]}"

# ---------------------------------------------------------------- stage 1: PTQ
for TIER in w4a16 w4a4; do
  DEST="$CKPT_ROOT/$TIER"
  if [[ -f "$DEST/ptq_meta.json" ]]; then
    echo "[nvfp4] $TIER checkpoint already present, skipping PTQ"
    continue
  fi
  echo "=============== PTQ $TIER ==============="
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "$ROOT/scripts/ptq_nvfp4.py" \
    --model "$MODEL" --tier "$TIER" --out "$DEST" \
    --calib-data "$CALIB" --calib-samples "$CALIB_SAMPLES" \
    2>&1 | tee "/historical/shared/qat/r/nvfp4/ptq-$TIER-${SLURM_JOB_ID}.log"
  if [[ ! -f "$DEST/ptq_meta.json" ]]; then
    echo "[nvfp4] PTQ $TIER failed; not evaluating" >&2
    exit 1
  fi
done

# ------------------------------------------------------------- stage 2: evals
# label|model|kv_cache_dtype
# The full 2x2 over {weights/acts} x {KV}, so a collapse can be attributed to
# one factor rather than to the combination.  w16a16kv4 is the decisive cell:
# unquantised weights with an NVFP4 KV cache isolates the KV path alone.
# label|model|kv_cache_dtype|extra_args
#
# Every KV-quantised tier is run with --disable-q-quantization.  The query is
# computed fresh each step and never enters the cache, so compressing it saves
# no memory; vLLM does it only because the trtllm-gen kernel that reads a 4-bit
# KV cache wants a low-precision query to pair with. That is a compute-precision
# change the --kv-cache-dtype flag does not name, and it does not belong in the
# KV tier's ledger.  The *_qq rows keep the default path for comparison.
ALL_SPECS=(
  "bf16|$MODEL|auto|"
  "w4a16kv16|$CKPT_ROOT/w4a16|auto|"
  "w4a4kv16|$CKPT_ROOT/w4a4|auto|"
  "w16a16kv4|$MODEL|nvfp4|--disable-q-quantization"
  "w4a16kv4|$CKPT_ROOT/w4a16|nvfp4|--disable-q-quantization"
  "w4a4kv4|$CKPT_ROOT/w4a4|nvfp4|--disable-q-quantization"
  "w16a16kv4_qq|$MODEL|nvfp4|"
  "w4a16kv4_qq|$CKPT_ROOT/w4a16|nvfp4|"
  "w4a4kv4_qq|$CKPT_ROOT/w4a4|nvfp4|"
  # FP8 KV: what four of NVIDIA's five production PTQ recipes actually ship.
  # Unlike nvfp4 it is a real torch dtype, so the no-q-quantisation variant
  # should not hit the getattr(torch, "nvfp4") crash and the two effects can
  # finally be separated.
  "w4a16kv8|$CKPT_ROOT/w4a16|fp8_e4m3|--disable-q-quantization"
  "w4a16kv8_qq|$CKPT_ROOT/w4a16|fp8_e4m3|"
  "w16a16kv8|$MODEL|fp8_e4m3|--disable-q-quantization"
  "w16a16kv8_qq|$MODEL|fp8_e4m3|"
  # KV precision sweep at BF16 weights, so the KV tier is the only variable.
  # k3v4 exists from earlier work but at avg@16 / TP=2; TP alone moved this
  # benchmark 1.67 pp, so it is re-measured here rather than compared across
  # protocols.
  "kv_int8pth|$MODEL|int8_per_token_head|"
  "kv_fp8pth|$MODEL|fp8_per_token_head|"
  "kv_tq4bit|$MODEL|turboquant_4bit_nc|"
  "kv_tqk3v4|$MODEL|turboquant_k3v4_nc|"
  "kv_tq3bit|$MODEL|turboquant_3bit_nc|"
  # 2-bit tiers.  Keys carry the Lloyd-Max codebook and values a uniform
  # grid, so values lose more at a fixed width -- k2v3 spends the extra
  # bit where it buys most, the same reasoning behind the existing k3v4.
  "kv_tqk2v3|$MODEL|turboquant_k2v3_nc|"
  "kv_tq2bit|$MODEL|turboquant_2bit_nc|"
  # Calibrated counterparts.  vLLM loads k_scale/v_scale from the checkpoint
  # and otherwise falls back to 1.0; this is a plain BF16 HF checkpoint, so
  # every row above that needs calibration was measured at scale=1.0.  Their
  # outputs repeat tokens ("OkayOkay", "Let Let", "minutes minutes") rather
  # than degrading smoothly, which is what a badly scaled cache looks like --
  # not evidence that the dtype itself is unusable.  TurboQuant needs no
  # calibration (Hadamard rotation plus a fixed Lloyd-Max codebook), so it has
  # no calibrated variant and none of these rows change its numbers.
  "kv_int8pth_cal|$MODEL|int8_per_token_head|--calculate-kv-scales"
  "kv_fp8pth_cal|$MODEL|fp8_per_token_head|--calculate-kv-scales"
  "w16a16kv8_cal|$MODEL|fp8_e4m3|--disable-q-quantization --calculate-kv-scales"
  "w16a16kv4_cal|$MODEL|nvfp4|--disable-q-quantization --calculate-kv-scales"
  # nvfp4 with the query left quantised.  --disable-q-quantization sends
  # prefill through plan(kv_data_type="nvfp4") -> getattr(torch, "nvfp4"),
  # which raises AttributeError; that path is still broken in 0.28, so this
  # is the only nvfp4 row on which calibration can actually be measured.
  "w16a16kv4_qq_cal|$MODEL|nvfp4|--calculate-kv-scales"
)
# TIERS selects a subset; default is everything.
TIERS="${TIERS:-bf16,w4a16kv16,w4a4kv16,w16a16kv4,w4a16kv4,w4a4kv4}"
SPECS=()
for spec in "${ALL_SPECS[@]}"; do
  label="${spec%%|*}"
  [[ ",$TIERS," == *",$label,"* ]] && SPECS+=("$spec")
done
echo "[nvfp4] evaluating tiers: ${TIERS}  (${#SPECS[@]} configs)"

if [[ ${#SPECS[@]} -gt ${#GPUS[@]} ]]; then
  echo "[nvfp4] ${#SPECS[@]} tiers but only ${#GPUS[@]} free GPU(s); rerun with fewer tiers" >&2
  exit 3
fi

pids=()
for i in "${!SPECS[@]}"; do
  IFS='|' read -r LABEL SRC KV EXTRA <<< "${SPECS[$i]}"
  echo "[nvfp4] eval $LABEL  model=$SRC  kv=$KV  extra='${EXTRA}'  gpu=${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python "$ROOT/scripts/eval_vllm_turboquant.py" \
    --model "$SRC" \
    --benchmark aime_math \
    --data "$DATA" \
    --n "$N" \
    --kv-cache-dtype "$KV" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $EXTRA \
    --output "$OUT_DIR/eval/$LABEL.json" \
    > "$OUT_DIR/eval/$LABEL.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "=============== results ==============="
python - "$OUT_DIR/eval" <<'PY'
import json, sys
from pathlib import Path
order = ["bf16", "w4a16kv16", "w4a4kv16",
         "w4a16kv8", "w4a16kv8_qq", "w16a16kv8", "w16a16kv8_qq",
         "w16a16kv4", "w4a16kv4", "w4a4kv4",
         "w16a16kv4_qq", "w4a16kv4_qq", "w4a4kv4_qq",
         "kv_int8pth", "kv_fp8pth", "kv_tq4bit", "kv_tqk3v4", "kv_tq3bit", "kv_tqk2v3", "kv_tq2bit"]
base = None
for label in order:
    f = Path(sys.argv[1]) / f"{label}.json"
    if not f.exists():
        print(f"  {label:12s} MISSING")
        continue
    d = json.loads(f.read_text())
    acc = d["accuracy"] * 100
    if label == "bf16":
        base = acc
    delta = "" if base is None or label == "bf16" else f"  ({acc - base:+.2f} vs bf16)"
    print(f"  {label:12s} {acc:6.2f}%  n={d['sampling']['n']}  "
          f"tp={d['tensor_parallel_size']}  "
          f"mean_resp={d.get('mean_response_tokens', 0):.0f}  "
          f"trunc={d.get('truncated_fraction', 0) * 100:.1f}%{delta}")
PY

[[ $fail -eq 0 ]] || { echo "[nvfp4] at least one eval failed" >&2; exit 1; }
echo "[nvfp4] done -> $OUT_DIR/eval"
