#!/usr/bin/env bash
# Watch one 1.5B QAT run and submit exactly one AIME2024 avg@4 evaluation
# for each requested checkpoint.  No step-0, validation, baseline, or extra
# benchmark jobs are created here.
set -euo pipefail

: "${RUN_DIR:?set RUN_DIR}"
: "${ARM:?set ARM (opd_only or nmse_only)}"
STEPS="${STEPS:-30,60,90,120,150}"
N="${N:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ROOT=/historical/home/QAT

IFS=',' read -r -a STEP_LIST <<< "$STEPS"

while :; do
  all_done=1
  for step in "${STEP_LIST[@]}"; do
    output="$RUN_DIR/eval/turboquant_k3v4_nc_step${step}.json"
    marker="$RUN_DIR/eval/.submitted_${step}"
    if [[ -s "$output" ]]; then
      continue
    fi
    all_done=0

    if [[ -f "$marker" ]]; then
      job=$(<"$marker")
      state=$(squeue -h -j "$job" -o '%T' 2>/dev/null | head -n 1 || true)
      if [[ -z "$state" ]]; then
        state=$(sacct -X -n -j "$job" --format=State 2>/dev/null | tr -d ' ' | head -n 1 || true)
      fi
      case "$state" in
        PENDING|RUNNING|CONFIGURING|COMPLETING)
          continue
          ;;
        COMPLETED)
          [[ -s "$output" ]] && continue
          ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
          rm -f -- "$marker"
          ;;
        *)
          continue
          ;;
      esac
    fi

    checkpoint="$RUN_DIR/checkpoints/global_step_${step}/actor"
    [[ -s "$checkpoint/fsdp_config.json" ]] || continue

    mkdir -p "$RUN_DIR/eval"
    export_args="ALL,RUN_DIR=$RUN_DIR,STEPS=$step,KV_DTYPE=turboquant_k3v4_nc,N=$N,MAX_NUM_SEQS=$MAX_NUM_SEQS,TAG=${ARM}"
    if job=$(sbatch --parsable --export="$export_args" \
        "$ROOT/scripts/slurm_qat_checkpoint_eval.sbatch" 2>"$RUN_DIR/eval/submit_step${step}.err"); then
      printf '%s\n' "$job" > "$marker"
      printf '%s step=%s job=%s\n' "$(date '+%F %T %Z')" "$step" "$job" \
        >> "$RUN_DIR/eval/submitted.log"
      rm -f -- "$RUN_DIR/eval/submit_step${step}.err"
    fi
  done

  if (( all_done )); then
    exit 0
  fi
  sleep 60
done
