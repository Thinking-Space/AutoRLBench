CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_20260921_164910/global_step_100 \
    --served-model-name global_step_100 \
    --host 127.0.0.1 \
    --port 8218 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_20260921_164910/global_step_693 \
    --served-model-name global_step_693 \
    --host 127.0.0.1 \
    --port 8219 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code



/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8218/v1 \
    --model global_step_100 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn \
    --enable-thinking

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8218/v1 \
    --model global_step_100 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn 

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8219/v1 \
    --model global_step_693 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn \
    --enable-thinking

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8219/v1 \
    --model global_step_693 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn 