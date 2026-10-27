#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODELS_ROOT="${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"
# shellcheck source=../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

PIPELINE_CONDA_ENV="${PIPELINE_CONDA_ENV:-${CONDA_ENV:-skillrl}}"
if [[ -n "$PIPELINE_CONDA_ENV" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$PIPELINE_CONDA_ENV"
    set -u
fi

cd "$PROJECT_ROOT"

NUM_TASKS="${NUM_TASKS:-180}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
EXPECTED_RECORDS=$((NUM_TASKS * ROLLOUTS_PER_TASK))
PIPELINE_BASE_NAME="${PIPELINE_BASE_NAME:-ezpoints_episode_skill_pipeline_qwen25_vl_3b_${SFT_SELF_DIR_SUFFIX}_${EXPECTED_RECORDS}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$PROJECT_ROOT/outputs/$PIPELINE_BASE_NAME}"
LOG_DIR="${LOG_DIR:-$PIPELINE_ROOT/logs}"
mkdir -p "$PIPELINE_ROOT" "$LOG_DIR"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-VL-3B-Instruct}"
BASELINE_ROLLOUTS_JSONL="${BASELINE_ROLLOUTS_JSONL:-$PIPELINE_ROOT/baseline_rollouts.jsonl}"
CANDIDATE_SKILLS_JSONL="${CANDIDATE_SKILLS_JSONL:-$PIPELINE_ROOT/candidate_skills.jsonl}"
BASELINE_IMAGE_ROOT="${BASELINE_IMAGE_ROOT:-$PIPELINE_ROOT/ezpoints_images}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

existing_rollout_count=0
if [[ -f "$BASELINE_ROLLOUTS_JSONL" ]]; then
    existing_rollout_count="$(wc -l < "$BASELINE_ROLLOUTS_JSONL")"
fi

if [[ "${SKIP_BASELINE_ROLLOUTS:-false}" == "true" ]]; then
    log "Stage 1/3: baseline collection explicitly skipped."
elif (( existing_rollout_count == EXPECTED_RECORDS )) && [[ "${RECOLLECT_BASELINE_ROLLOUTS:-false}" != "true" ]]; then
    log "Stage 1/3: reusing $existing_rollout_count complete baseline rollouts."
else
    log "Stage 1/3: collecting ${NUM_TASKS} x ${ROLLOUTS_PER_TASK} visual EZPoints baseline rollouts."
    env \
        MODELS_ROOT="$MODELS_ROOT" \
        MODEL_PATH="$BASE_MODEL_PATH" \
        OUTPUT_DIR="$PIPELINE_ROOT" \
        BASELINE_ROLLOUTS_JSONL="$BASELINE_ROLLOUTS_JSONL" \
        BASELINE_CONDA_ENV="$PIPELINE_CONDA_ENV" \
        NUM_TASKS="$NUM_TASKS" \
        ROLLOUTS_PER_TASK="$ROLLOUTS_PER_TASK" \
        TASK_BATCH_SIZE="${BASELINE_TASK_BATCH_SIZE:-180}" \
        REQUEST_WORKERS="${BASELINE_REQUEST_WORKERS:-64}" \
        MAX_STEPS="${BASELINE_MAX_STEPS:-8}" \
        SEED="${BASELINE_SEED:-2026}" \
        POLICY_TEMPERATURE="${BASELINE_POLICY_TEMPERATURE:-1.0}" \
        POLICY_MAX_COMPLETION_TOKENS="${BASELINE_POLICY_MAX_COMPLETION_TOKENS:-512}" \
        DATA_PARALLEL_SIZE="${BASELINE_DATA_PARALLEL_SIZE:-8}" \
        PORT="${BASELINE_POLICY_PORT:-60004}" \
        RESUME="${BASELINE_RESUME:-true}" \
        OVERWRITE="${BASELINE_OVERWRITE:-false}" \
        bash "$SCRIPT_DIR/_collect_rollouts.sh" \
        2>&1 | tee "$LOG_DIR/baseline_rollouts.log"
fi

if [[ ! -f "$BASELINE_ROLLOUTS_JSONL" ]]; then
    log "Baseline rollout file not found: $BASELINE_ROLLOUTS_JSONL"
    exit 1
fi
actual_rollout_count="$(wc -l < "$BASELINE_ROLLOUTS_JSONL")"
if (( actual_rollout_count != EXPECTED_RECORDS )); then
    log "Expected $EXPECTED_RECORDS baseline rollouts, found $actual_rollout_count."
    exit 1
fi

SKILL_MODEL="${EZPOINTS_SKILL_MODEL:-${SKILL_MODEL:-${OPENAI_MODEL:-}}}"
SKILL_BASE_URL="${EZPOINTS_SKILL_BASE_URL:-${OPENAI_BASE_URL:-}}"
SKILL_API_KEY="${EZPOINTS_SKILL_API_KEY:-${OPENAI_API_KEY:-}}"
if [[ -z "$SKILL_MODEL" ]]; then
    log "Set OPENAI_MODEL or EZPOINTS_SKILL_MODEL for episode-skill generation."
    exit 1
fi

if [[ "$SFT_TEACHER_SHORT" == "gemini" ]]; then
    DEFAULT_SKILL_MAX_COMPLETION_TOKENS=8192
    DEFAULT_SKILL_PARSE_ATTEMPTS=3
else
    DEFAULT_SKILL_MAX_COMPLETION_TOKENS=2048
    DEFAULT_SKILL_PARSE_ATTEMPTS=2
fi

log "Stage 2/3: generating $EXPECTED_RECORDS EZPoints episode skills with $SKILL_MODEL."
skill_args=(
    --baseline-rollouts "$BASELINE_ROLLOUTS_JSONL"
    --output-dir "$PIPELINE_ROOT"
    --skill-gen-workers "${EZPOINTS_SKILL_GEN_WORKERS:-64}"
    --skill-model "$SKILL_MODEL"
    --skill-max-completion-tokens "${EZPOINTS_SKILL_MAX_COMPLETION_TOKENS:-$DEFAULT_SKILL_MAX_COMPLETION_TOKENS}"
    --skill-parse-attempts "${EZPOINTS_SKILL_PARSE_ATTEMPTS:-$DEFAULT_SKILL_PARSE_ATTEMPTS}"
)
if [[ -n "$SKILL_BASE_URL" ]]; then
    skill_args+=(--skill-base-url "$SKILL_BASE_URL")
fi
if [[ -n "$SKILL_API_KEY" ]]; then
    skill_args+=(--skill-api-key "$SKILL_API_KEY")
fi
if [[ "${EZPOINTS_INCLUDE_EPISODE_SUMMARY:-true}" != "true" ]]; then
    skill_args+=(--no-include-episode-summary)
fi
if [[ "${EZPOINTS_SKILL_RESUME:-true}" == "true" ]]; then
    skill_args+=(--resume)
fi
if [[ "${EZPOINTS_REGENERATE_CANDIDATES:-false}" == "true" ]]; then
    skill_args+=(--regenerate-candidates)
fi
if [[ "${EZPOINTS_MAX_CANDIDATES:-null}" != "null" ]]; then
    skill_args+=(--max-candidates "$EZPOINTS_MAX_CANDIDATES")
fi

python3 "$SCRIPT_DIR/generate_candidate_skills.py" "${skill_args[@]}" \
    2>&1 | tee "$LOG_DIR/generate_candidate_skills.log"

if [[ ! -f "$CANDIDATE_SKILLS_JSONL" ]]; then
    log "Candidate skill file not found: $CANDIDATE_SKILLS_JSONL"
    exit 1
fi

log "Stage 3/3: building EZPoints visual episode-skill SFT data."
build_args=(
    --rollout-dir "$PIPELINE_ROOT"
    --image-root "$BASELINE_IMAGE_ROOT"
    --output-dir "$PIPELINE_ROOT"
    --candidate-skills "$CANDIDATE_SKILLS_JSONL"
    --val-ratio "${SFT_VAL_RATIO:-0.1}"
    --seed "${SFT_DATA_SEED:-2026}"
    --max-records "${SFT_MAX_RECORDS:-0}"
)
if [[ "${EZPOINTS_INCLUDE_EPISODE_SUMMARY:-true}" != "true" ]]; then
    build_args+=(--no-include-episode-summary)
fi

python3 "$SCRIPT_DIR/build_sft_from_rollouts.py" "${build_args[@]}" \
    2>&1 | tee "$LOG_DIR/build_sft.log"

SUMMARY_FILE="$PIPELINE_ROOT/data_pipeline_summary.env"
{
    printf 'PIPELINE_ROOT=%s\n' "$PIPELINE_ROOT"
    printf 'TEACHER_MODEL=%s\n' "$SKILL_MODEL"
    printf 'BASELINE_ROLLOUTS_JSONL=%s\n' "$BASELINE_ROLLOUTS_JSONL"
    printf 'CANDIDATE_SKILLS_JSONL=%s\n' "$CANDIDATE_SKILLS_JSONL"
    printf 'SFT_TRAIN_DATA=%s\n' "$PIPELINE_ROOT/sft_episode_skill_train.parquet"
    printf 'SFT_VAL_DATA=%s\n' "$PIPELINE_ROOT/sft_episode_skill_val.parquet"
} > "$SUMMARY_FILE"
log "EZPoints SFT data pipeline finished: $SUMMARY_FILE"
