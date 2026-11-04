CUDA_VISIBLE_DEVICES=0 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8218 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=1 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8219 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=2 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8220 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=3 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8221 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=4 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8222 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=5 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8223 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=6 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8224 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code

CUDA_VISIBLE_DEVICES=7 /local/user/conda_envs/vllm_qwen_1/bin/vllm serve \
  /local/user/Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8225 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --trust-remote-code