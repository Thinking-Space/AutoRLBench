
#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
assets_dir="${SEARCH_R1_ASSETS_DIR:-$(dirname -- "$repo_dir")}"
default_runtime_env="$assets_dir/.venv-search-r1"
if [[ -x "$repo_dir/runtime/train/.venv/bin/python" ]]; then
    default_runtime_env="$repo_dir/runtime/train/.venv"
fi
runtime_env="${SEARCH_R1_RUNTIME_ENV:-$default_runtime_env}"
base_env="${SEARCH_R1_BASE_ENV:-/mnt/afs/L202500003/miniconda3/envs/vllm}"
index_file="${SEARCH_R1_INDEX:-$assets_dir/search_r1_e5_fixture/e5_Flat.index}"
corpus_file="${SEARCH_R1_CORPUS:-$assets_dir/search_r1_e5_fixture/corpus.jsonl}"
retriever_path="${SEARCH_R1_MODEL:-$assets_dir/models/e5-base-v2}"

if [[ -f "$assets_dir/faiss-build/faiss/python/faiss/__init__.py" ]]; then
    export PYTHONPATH="$assets_dir/faiss-build/faiss/python${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$assets_dir/.venv-faiss/lib/python3.12/site-packages/faiss_gpu_cu12.libs:$base_env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$base_env/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec "$runtime_env/bin/python" -u "$repo_dir/search_r1/search/retrieval_server.py" \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk "${SEARCH_R1_TOPK:-5}" \
    --retriever_name e5 \
    --retriever_model "$retriever_path" "$@"
