#!/usr/bin/env bash
# Submit the requested external benchmark evaluation as each VeRL checkpoint
# becomes a complete Hugging Face export.  This watcher uses a five-minute
# cadence and never touches the trainer process or its GPUs.
set -euo pipefail

# DEPRECATED -- refuses to run without QAT_WATCHER_I_KNOW=1.
#
# Two defects, both hit on 2026-09-12 (jobs 24045-24137):
#
#  1. It takes its parameters from profile_env.py, and every 1.5B run in this
#     repository is launched with --profile deepseek_skywork (the 7B profile).
#     So BASE_MODEL was the 7B, and each checkpoint merged 1.5B tensors into a
#     7B config.json.  vLLM then died in WorkerProc init.
#  2. On FAILED/CANCELLED/TIMEOUT it deletes the marker and resubmits on the
#     next five-minute poll, with no failure counter and no backoff.  Combined
#     with (1) that is an unbounded loop: 46 submissions, every one dead within
#     four minutes, each holding two GPUs.
#
# Use experiments/2026-09-11/scripts/eval_kvqat_manual.sbatch instead: one job
# per checkpoint, submitted by hand, parameters read from an explicitly named
# profile, and the merged config.json verified before the engine is built.
if [[ "${QAT_WATCHER_I_KNOW:-0}" != "1" ]]; then
  echo "watch_mopd_checkpoints_eval.sh is deprecated; see the comment at the top." >&2
  echo "Use experiments/2026-09-11/scripts/eval_kvqat_manual.sbatch." >&2
  exit 64
fi


: "${PROFILE:?set PROFILE=deepseek_skywork or qwen25_longrlvr}"
ROOT=/historical/home/QAT
# The scheme the checkpoints were trained for.  Evaluating a k2v3 student
# through a 3-bit cache measures neither the student nor the scheme.
METHOD="${METHOD:-turboquant_3bit_nc}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
RUN_ROOT="${RUN_ROOT:-/historical/shared/qat/runs/mopd/${PROFILE}/kv32/main200}"
# Overridable: a run with save_freq=30 checkpoints at 30,60,... not 20,40,...
IFS=' ' read -r -a CHECKPOINTS <<< "${CHECKPOINTS:-20 40 60 80 100 120 140 160 180 200}"
ENV_DIR=/historical/shared/bitdistiller/envs/vllm-0.19.1-test
# The vLLM environment intentionally does not include the optional
# ``accelerate`` package used by VeRL's official FSDP checkpoint merger.
# Keep the merger dependency isolated so its older accelerate installation
# does not override vLLM's transformers/torch packages.
MERGER_DEPS=/historical/shared/qat/tmp/merger-deps-v2-20260829
MERGER_OVERLAY=/historical/shared/qat/tmp/turboquant-train-overlay
if [[ ! -s "$MERGER_DEPS/accelerate/__init__.py" ]]; then
  echo "missing merger dependency: $MERGER_DEPS/accelerate" >&2
  exit 1
fi

merged_ready() {
  [[ -s "$1/model.safetensors.index.json" || -s "$1/model.safetensors" \
     || -s "$1/pytorch_model.bin.index.json" || -s "$1/pytorch_model.bin" ]]
}

# The profile table in preflight_mopd.py is the single definition of a
# profile.  This block used to be a private copy, which is why a 1.5B run
# launched here inherited the 7B's benchmark window, concurrency and
# tensor-parallel width.  Explicit environment values still win.
_env=$("$ENV_DIR/bin/python" "$ROOT/scripts/profile_env.py" "$PROFILE") || exit 2
while IFS='=' read -r _k _v; do
  [[ -n "$_k" ]] || continue
  # Only fill in what the caller did not set.
  [[ -n "${!_k:-}" ]] || eval "$_k=$_v"
done <<< "$_env"
export BENCHMARK BASE_MODEL DATA MAX_MODEL_LEN MAX_NUM_SEQS TENSOR_PARALLEL_SIZE N MAX_NUM_BATCHED_TOKENS

while :; do
  all_done=1
  for step in "${CHECKPOINTS[@]}"; do
    checkpoint="$RUN_ROOT/checkpoints/global_step_${step}/actor"
    model="$RUN_ROOT/merged_step${step}"
    marker="$RUN_ROOT/eval/.submitted_${step}"
    if [[ -f "$marker" ]]; then
      submitted_job=$(<"$marker")
      state=$(squeue -h -j "$submitted_job" -o '%T' 2>/dev/null | head -n 1 || true)
      if [[ -z "$state" ]]; then
        state=$(sacct -X -n -j "$submitted_job" --format=State 2>/dev/null | tr -d ' ' | head -n 1 || true)
      fi
      case "$state" in
        PENDING|RUNNING|CONFIGURING|COMPLETING)
          continue
          ;;
        COMPLETED)
          [[ -s "$RUN_ROOT/eval/${BENCHMARK}-${METHOD}-step${step}.json" ]] && continue
          ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
          # Allow the next five-minute poll to resubmit a failed checkpoint.
          rm -f "$marker"
          ;;
        *)
          # Unknown accounting state: keep the submission marker and avoid a
          # duplicate while Slurm finishes publishing its state.
          continue
          ;;
      esac
    fi
    all_done=0
    # VeRL stores sharded FSDP tensors.  Use its official merger before
    # handing the checkpoint to vLLM; the tokenizer-only huggingface folder
    # is not itself a loadable model.
    if [[ ! -s "$checkpoint/fsdp_config.json" ]]; then
      continue
    fi
    mkdir -p "$RUN_ROOT/eval"
    if ! merged_ready "$model"; then
      if ! PYTHONPATH="$MERGER_DEPS:$MERGER_OVERLAY:/historical/shared/qat/tmp/reference/Open-MOPD/training/verl:/historical/shared/qat/envs/mopd-pylibs:${PYTHONPATH:-}" \
          "$ENV_DIR/bin/python" \
          "/historical/shared/qat/tmp/reference/Open-MOPD/training/verl/scripts/legacy_model_merger.py" \
          merge --backend fsdp --local_dir "$checkpoint" \
          --hf_model_path "$BASE_MODEL" --target_dir "$model" \
          >> "$RUN_ROOT/eval/merge.log" 2>&1; then
        continue
      fi
    fi
    if ! merged_ready "$model"; then
      continue
    fi
    submit_err="$RUN_ROOT/eval/submit_step${step}.err"
    export_args="ALL,N=$N,TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE,BENCHMARK=$BENCHMARK,METHOD=$METHOD,MODEL=$model,OUTPUT=$RUN_ROOT/eval/${BENCHMARK}-${METHOD}-step${step}.json,DATA=$DATA,MAX_MODEL_LEN=$MAX_MODEL_LEN,MAX_NUM_SEQS=$MAX_NUM_SEQS,MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS,GPU_MEMORY_UTILIZATION=0.90"
    if ! job=$(sbatch --parsable \
      --export="$export_args" \
      "$ROOT/scripts/slurm_eval_turboquant.sbatch" 2>"$submit_err"); then
      printf '%s step=%s submit_failed: %s\n' "$(date '+%F %T %Z')" "$step" \
        "$(tr '\n' ' ' < "$submit_err")" >> "$RUN_ROOT/eval/submitted.log"
      continue
    fi
    rm -f "$submit_err"
    printf '%s\n' "$job" > "$marker"
    printf '%s step=%s job=%s\n' "$(date '+%F %T %Z')" "$step" "$job" \
      >> "$RUN_ROOT/eval/submitted.log"
  done
  if (( all_done )); then
    exit 0
  fi
  sleep 300
done
