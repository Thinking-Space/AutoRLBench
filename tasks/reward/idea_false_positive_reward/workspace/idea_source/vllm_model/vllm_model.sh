export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
vllm serve /local/user/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 2398

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
vllm serve /local/user/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 2399

CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_20260914_100457/global_step_693 \
    --served-model-name global_step_693 \
    --host 127.0.0.1 \
    --port 8218 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

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

CUDA_VISIBLE_DEVICES=0 /local/user/conda_envs/seed/bin/vllm serve \
  /local/user/models/sft/awm_episode_skill_sft_qwen3_8b_ep3_20260914_100457/global_step_693 \
  --served-model-name global_step_693 \
  --host 127.0.0.1 --port 8218 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=1 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8219 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=1 /local/user/conda_envs/vllm_qwen_3_8/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_715 \
  --served-model-name global_step_715 \
  --host 127.0.0.1 --port 8219 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /local/user/models/opsd-tool/awm_episode_skill_sft_qwen3_8b_ep3_20260921_133147/global_step_1 \
    --served-model-name global_step_1 \
    --host 127.0.0.1 \
    --port 8220 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=3 /local/user/conda_envs/vllm_qwen_3_8/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_1430 \
  --served-model-name global_step_1430 \
  --host 127.0.0.1 --port 8221 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=4 /local/user/conda_envs/vllm_qwen_3_8/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_2145 \
  --served-model-name global_step_2145 \
  --host 127.0.0.1 --port 8222 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=5 /local/user/conda_envs/vllm_qwen_3_8/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_2145 \
  --served-model-name global_step_2145 \
  --host 127.0.0.1 --port 8223 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=0 /local/user/conda_envs/vllm_qwen_3_8/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_1430 \
  --served-model-name global_step_1430 \
  --host 127.0.0.1 --port 8218 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=1 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /workspace/idea_source/outputs/qwen3_8b_awm_skill_sft_verl_structured/global_step_1 \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name global_step_1 \
  --host 127.0.0.1 --port 8219 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=2 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /workspace/idea_source/outputs/qwen3_8b_awm_skill_sft_verl_structured/global_step_100 \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name global_step_100 \
  --host 127.0.0.1 --port 8220 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=3 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /workspace/idea_source/outputs/qwen3_8b_awm_skill_sft_verl_structured/global_step_735 \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name global_step_735 \
  --host 127.0.0.1 --port 8221 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=4 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_2145 \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b-awm-natural-step-2145 \
  --host 127.0.0.1 --port 8222 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code


CUDA_VISIBLE_DEVICES=5 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8223 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=6 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/models/qwen3_8b_awm_skill_sft_verl_12288_natural/global_step_100 \
  --tokenizer /local/user/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b-awm-natural-step-100 \
  --host 127.0.0.1 --port 8224 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8218/v1 \
    --model global_step_19 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn \
    --enable-thinking

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8218/v1 \
    --model global_step_19 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn 

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8219/v1 \
    --model Qwen3-8B \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn \
    --enable-thinking

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8219/v1 \
    --model Qwen3-8B \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn 

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8220/v1 \
    --model global_step_100 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn \
    --enable-thinking

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8220/v1 \
    --model global_step_100 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn 

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8221/v1 \
    --model global_step_735 \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn.py \
    --api-url http://127.0.0.1:8223/v1 \
    --model Qwen3-8B \
    --output-dir /workspace/idea_source/outputs/awm_debug_appointment_second_turn_qwen3 

/workspace/awm_source/.venv/bin/python \
    /workspace/idea_source/scripts/sft/awm/debug_appointment_second_turn_flattened.py \
    --api-url http://127.0.0.1:8223/v1 \
    --model Qwen3-8B