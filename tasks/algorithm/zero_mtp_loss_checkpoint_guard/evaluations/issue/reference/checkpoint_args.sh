#!/usr/bin/env bash

# Build Training actor/reference checkpoint arguments without touching Miles core.
training_build_checkpoint_args() {
  # A train-only smoke deliberately runs one rollout and saves its final state.
  if [ "${DEBUG_TRAIN_ONLY:-0}" != "1" ] && \
     [ -n "${NUM_ROLLOUT:-}" ] && [ "${SAVE_INTERVAL:-20}" -ge "${NUM_ROLLOUT}" ]; then
    echo "ERROR: SAVE_INTERVAL=${SAVE_INTERVAL:-20} >= NUM_ROLLOUT=${NUM_ROLLOUT};" \
         "the run would produce only the forced final-rollout checkpoint." >&2
    return 1
  fi
  CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --save "${CKPT_PATH}/${EXP_NAME}/"
    # Unit is outer rollouts, not optimizer steps: train_async.py passes
    # rollout_id into should_run_periodic_action. A value above NUM_ROLLOUT
    # leaves the forced final-rollout save as the only checkpoint.
    --save-interval "${SAVE_INTERVAL:-20}"
    --ref-load "${MEGATRON_MODEL_PATH}"
  )

  if [ -n "${TRAINING_RESUME_CKPT:-}" ]; then
    if [ ! -d "${TRAINING_RESUME_CKPT}" ]; then
      echo "ERROR: TRAINING_RESUME_CKPT is not a directory: ${TRAINING_RESUME_CKPT}" >&2
      return 1
    fi
    if [ ! -f "${TRAINING_RESUME_CKPT}/latest_checkpointed_iteration.txt" ] && \
       [ -z "${TRAINING_RESUME_CKPT_STEP:-}" ]; then
      echo "ERROR: resume checkpoint needs latest_checkpointed_iteration.txt or TRAINING_RESUME_CKPT_STEP" >&2
      return 1
    fi

    CKPT_ARGS+=(--load "${TRAINING_RESUME_CKPT}")
    CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
    if [ -n "${TRAINING_RESUME_CKPT_STEP:-}" ]; then
      CKPT_ARGS+=(--ckpt-step "${TRAINING_RESUME_CKPT_STEP}")
    fi
    if [ -n "${MEGATRON_CKPT_STEP:-}" ]; then
      CKPT_ARGS+=(--ref-ckpt-step "${MEGATRON_CKPT_STEP}")
    fi

    # Leave this unset so Miles restores loaded_rollout_id + 1 from the actor checkpoint.
    ROLLOUT_START_ARGS=()
    return 0
  fi

  CKPT_ARGS+=(
    --load "${MEGATRON_MODEL_PATH}"
    --finetune
    --no-load-optim
    --no-load-rng
  )
  if [ -n "${MEGATRON_CKPT_STEP:-}" ]; then
    CKPT_ARGS+=(--ckpt-step "${MEGATRON_CKPT_STEP}" --ref-ckpt-step "${MEGATRON_CKPT_STEP}")
  fi
  ROLLOUT_START_ARGS=(--start-rollout-id 0)
}
