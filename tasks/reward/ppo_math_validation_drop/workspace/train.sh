#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
out=${AUTORLBENCH_RESULTS:?}
model="$PWD/assets/model"
data="$PWD/assets/data"
mkdir -p "$out/ppo"
python3 -u -m verl.trainer.main_ppo --config-path "$PWD" --config-name ppo \
    "actor_rollout_ref.model.path=$model" "critic.model.path=$model" \
    "data.train_files=$data/train.parquet" \
    "data.val_files=[$data/AIME24.parquet,$data/AIME25.parquet,$data/AMC23.parquet]" \
    "reward.custom_reward_function.path=$PWD/reward.py" \
    "trainer.default_local_dir=$out/ppo/checkpoints" \
    "trainer.validation_data_dir=$out/ppo/validation" "hydra.run.dir=$out/ppo/hydra" \
    "trainer.total_training_steps=${PPO_STEPS:-1000}" \
    "trainer.resume_mode=${PPO_RESUME_MODE:-disable}" \
    ${PPO_RESUME_PATH:+"trainer.resume_from_path=$PPO_RESUME_PATH"} "$@"
