#!/bin/bash
#SBATCH --job-name=replay-chain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=03:00:00
#SBATCH --output=/historical/shared/qat/r/rollout/replay-chain-%j.out
#SBATCH --error=/historical/shared/qat/r/rollout/replay-chain-%j.err
set -uo pipefail

# Gate between the teacher rollout and the full off-policy (LLM-QAT-style) arm.
#
# Three steps, each of which must pass before the next runs:
#   1. build the prompt subset, verifying every prompt re-tokenizes to the ids
#      stored in the trajectory file
#   2. run two training steps through the replay path
#   3. check that the importance ratio is sane, then submit the 150-step run
#
# Step 3 is the point of the whole job.  The replay can be wired to the wrong
# sequences and still train happily -- shapes all match, nothing raises, loss
# goes down.  What that failure looks like is the student assigning low
# probability to tokens it is told it drew, i.e. a log-ratio far from zero.
# The student is initialised AT the teacher and differs only by KV
# quantisation, so |log pi_s - log pi_t| must be small.  If it is not, the
# 150-step run is not submitted.

ROOT=/historical/home/QAT
ENV=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/vendor/verl:/historical/shared/qat/tmp/turboquant-train-overlay:/historical/shared/qat/envs/mopd-pylibs"
PYINC=/historical/shared/qat/tmp/python-devel/root/usr/include/python3.12
export CPATH="$PYINC" C_INCLUDE_PATH="$PYINC" CPLUS_INCLUDE_PATH="$PYINC"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/historical/shared/qat/cache/tmp}"
export PYTHONUNBUFFERED=1

MODEL=/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl
ROLLOUT_DIR=/historical/shared/qat/data/rollouts/teacher-1p5b-9600
TRAJ="$ROLLOUT_DIR/trajectories.parquet"
SUBSET=/historical/shared/qat/data/processed/skywork_replay_subset_9600.parquet
SOURCE=/historical/shared/qat/data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet
# Beyond this the student is not scoring the teacher's own tokens and the
# replay is matching the wrong sequences.
MAX_ABS_LOG_RATIO=2.0

echo "=============================================================="
echo "[chain] step 1/3  build + verify the prompt subset"
echo "=============================================================="
if [[ ! -f "$TRAJ" ]]; then
  echo "[chain] missing $TRAJ -- the rollout did not produce a merged file" >&2
  exit 1
fi
python "$ROOT/scripts/make_replay_prompt_subset.py" \
  --trajectories "$TRAJ" \
  --source "$SOURCE" \
  --out "$SUBSET" \
  --tokenizer "$MODEL" || { echo "[chain] subset build failed" >&2; exit 1; }

echo
echo "=============================================================="
echo "[chain] step 2/3  two training steps through the replay"
echo "=============================================================="
SMOKE_LOG="/historical/shared/qat/r/rollout/replay-smoke-${SLURM_JOB_ID}.log"

# Same knobs as the on-policy arm (job 8162) so the comparison stays
# single-variable: reverse-KL objective, no block-align term, lr 1.5e-6,
# global batch 64, top-k 512 union.  Teacher entropy is logging only.
export MODEL_PATH="$MODEL" TEACHER_PATH="$MODEL" TRAIN_PATH="$SUBSET"
export QAT_PROBE_OBJECTIVE=kl
export QAT_BLOCK_ALIGN_STATIC_REF=1
export QAT_SKIP_TEACHER_ENTROPY=1
export QAT_TEACHER_REPLAY="$TRAJ"
export GPUS=4 GB=64 MINI=64 MB=1
export BLOCK_ALIGN_COEF=0.0 BLOCK_ALIGN_LAYER_EVERY=4 BLOCK_ALIGN_KEEP_FRACTION=0.125
export BLOCK_ALIGN_METRIC=nmse
export LR=1.5e-6 REWARD_TOP_K=512 TOP_K_STRATEGY=union REWARD_WEIGHT_MODE=student_p
export PROMPT_LENGTH=512 RESPONSE_LENGTH=16384
export PROJECT_NAME=qat-distill-1p5b SWANLAB_MODE=cloud

UPDATES=2 SAVE_FREQ=1000 TAG=replay-smoke \
  bash "$ROOT/scripts/slurm_qat_train.sbatch" > "$SMOKE_LOG" 2>&1
SMOKE_RC=$?
echo "[chain] smoke exit=$SMOKE_RC  log=$SMOKE_LOG"
if [[ $SMOKE_RC -ne 0 ]]; then
  echo "[chain] smoke run failed; NOT submitting the full arm" >&2
  tail -n 60 "$SMOKE_LOG" >&2
  exit 1
fi

echo
echo "=============================================================="
echo "[chain] step 3/3  check the importance ratio"
echo "=============================================================="
python - "$SMOKE_LOG" "$MAX_ABS_LOG_RATIO" <<'PY'
import re
import sys

log, limit = sys.argv[1], float(sys.argv[2])
text = open(log, errors="ignore").read()

num = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
found = {}
for key in ("log_ratio_mean", "log_ratio_absmean", "ratio_mean", "ratio_max"):
    hits = re.findall(rf"teacher_replay/{key}:{num}", text)
    if hits:
        found[key] = [float(h) for h in hits]

if not found:
    raise SystemExit(
        "no teacher_replay/* metrics in the smoke log.  The replay branch never "
        "ran: either generate_sequences was not patched, or the batch reached "
        "the reward block without teacher_rollout_log_probs."
    )

for key, values in found.items():
    print(f"  teacher_replay/{key}: {[round(v, 5) for v in values]}")

absmean = found.get("log_ratio_absmean")
if not absmean:
    raise SystemExit("log_ratio_absmean missing; cannot judge the replay")
worst = max(absmean)
if worst > limit:
    raise SystemExit(
        f"mean |log pi_student - log pi_teacher| = {worst:.4f} > {limit}.  The "
        "student is initialised at the teacher and differs only by KV "
        "quantisation, so this should be near zero; a large value means the "
        "replay is serving responses that do not belong to these prompts."
    )
print(f"\n  OK: worst mean |log ratio| = {worst:.4f} <= {limit}")
PY
CHECK_RC=$?
if [[ $CHECK_RC -ne 0 ]]; then
  echo "[chain] ratio check failed; NOT submitting the full arm" >&2
  exit 1
fi

echo
echo "=============================================================="
echo "[chain] submitting the 150-step off-policy arm"
echo "=============================================================="
export UPDATES=150 SAVE_FREQ=30 TAG=llmqat-offline-revkl
FULL_ID=$(sbatch --parsable --job-name=qat1p5b-offline-revkl \
  --gres=gpu:nvidia_l20d:4 --time=2-00:00:00 \
  --export=ALL "$ROOT/scripts/slurm_qat_train.sbatch")
echo "[chain] submitted full off-policy arm as job $FULL_ID"
echo "[chain] done"
