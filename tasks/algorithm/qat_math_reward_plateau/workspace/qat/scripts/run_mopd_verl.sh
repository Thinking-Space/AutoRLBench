#!/usr/bin/env bash
# Launch the pinned Open-MOPD/veRL trainer for the QAT experiments.
#
# The wrapper is intentionally dry-run by default.  It delegates profile
# resolution, path/dependency checks, safe command printing, and optional
# JSONL->Parquet materialization to preflight_mopd.py.  The trainer itself is
# *not* reimplemented here: the command points at
# /historical/shared/qat/tmp/reference/Open-MOPD/training/verl and uses its native
# token_reward_direct + compute_distillation_reward path.
#
# Examples:
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork --run
#   bash scripts/run_mopd_verl.sh --profile qwen25_longrlvr --stage main \
#       --kv-bits 4 --require-eval-gate --require-quantized-rollout \
#       --eval-gate-summary /historical/shared/qat/runs/aime/kv-gate-qwen25_longrlvr/gate_selection.json \
#       --run
#
# The native TurboQuant path keeps QAT_KV_VLLM_ENABLE=0: vLLM itself owns the
# quantized cache and the rollout log-probs.  A native TurboQuant main run
# does not use the historical HF KV2/KV4 selector; the requested dtype is
# fixed by the experiment.  The older external-vLLM/HF-hook path remains
# fail-closed unless its runtime policy is explicitly verified:
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork \
#       --quantize-vllm-rollout --require-quantized-rollout --run
# Diagnostic actor-only main run (not end-to-end quantized rollout):
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork --stage main \
#       --allow-rollout-policy-mismatch --run
#
# The historical HF-hook main path can still be tied to the frozen-model
# bit-selection gate:
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork --stage main \
#       --kv-bits 4 --require-eval-gate \
#       --eval-gate-summary /historical/shared/qat/runs/aime/kv-gate-deepseek_skywork/gate_selection.json \
#       --run
# For a deliberately ungated actor-only diagnostic, omit the summary and use
# the explicit escape hatch below; a typo in a supplied summary path is still
# rejected:
#   bash scripts/run_mopd_verl.sh --profile deepseek_skywork --stage main \
#       --allow-missing-eval-gate --allow-rollout-policy-mismatch --run
#
# Actor storage defaults to FP32 (FSDP forward/reduction mixed precision is
# BF16/FP32/FP32); use --actor-model-dtype bf16 only as a deliberate memory
# diagnostic, because its optimizer-state precision is not claimed.
# HF bias preservation remains the cheaper post-forward approximation by
# default.  The extra-GEMM exact seam is explicit and still requires the
# preserve flag (it is not enabled by the profiles):
#   bash scripts/run_mopd_verl.sh --exact-bias-free-recompute ...
#
# No SWANLAB_API_KEY is put on the command line.  If the caller has already
# exported it, the Python helper passes the inherited environment through and
# only prints a redacted status.
# A fresh real run records a redacted effective_settings.json immediately
# before trainer startup; dry-runs never write it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MOPD_VENV_BIN is the one knob callers use to pick an environment, and the
# interpreter must follow it: Ray actors fork the driver, so an environment
# that reaches only PATH never reaches the workers.  Job 11825 launched
# int4_per_token_head against vLLM 0.21.0 for exactly this reason -- the
# sbatch put the 0.25.1 venv on PATH while this line stayed pinned to 0.21.0.
# MOPD_PREFLIGHT_PYTHON still wins, for deliberately splitting the two.
if [[ -n "${MOPD_PREFLIGHT_PYTHON:-}" ]]; then
    PYTHON_BIN="$MOPD_PREFLIGHT_PYTHON"
elif [[ -n "${MOPD_VENV_BIN:-}" ]]; then
    PYTHON_BIN="$MOPD_VENV_BIN/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "[run_mopd_verl] MOPD_VENV_BIN=$MOPD_VENV_BIN has no executable 'python';" >&2
        echo "[run_mopd_verl] refusing to fall back to another environment." >&2
        exit 2
    fi
else
    PYTHON_BIN=/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    # Keep the helper usable on a workstation where the cluster venv is not
    # mounted; the helper will report the selected interpreter as a preflight
    # warning rather than silently changing the training environment.
    PYTHON_BIN="${PYTHON:-python3}"
fi

export MOPD_PREFLIGHT_PYTHON="$PYTHON_BIN"
if [[ -n "${MOPD_VERL_DIR:-}" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/preflight_mopd.py" --verl-dir "$MOPD_VERL_DIR" "$@"
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/preflight_mopd.py" "$@"
