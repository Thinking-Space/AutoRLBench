CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8218 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8219 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8220 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8221 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8222 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=5 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8223 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=6 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8224 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

CUDA_VISIBLE_DEVICES=7 \
  PYTHONPATH=/workspace/idea_source/scripts \
  /local/user/conda_envs/seed/bin/python \
  /workspace/idea_source/scripts/vllm_idea_compat.py serve \
  /workspace/idea_source/outputs/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_19_hf \
    --served-model-name global_step_19 \
    --host 127.0.0.1 \
    --port 8225 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

