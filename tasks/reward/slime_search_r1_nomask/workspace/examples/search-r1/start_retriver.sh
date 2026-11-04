# Local dense retrieval server for the search rollout
# (generate_with_search.py posts to http://127.0.0.1:8000/retrieve).
# Same default as run_qwen2.5_3B.sh: the assets sit next to the working directory.
DATA_DIR=${DATA_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$PWD}

# The e5 encoder is loaded by name, so prefer a local copy and only fall back to
# the hub id when there is none -- loading it from the hub needs network.
RETRIEVER_MODEL=${RETRIEVER_MODEL:-$MODEL_DIR/e5-base-v2}
[ -d "$RETRIEVER_MODEL" ] || RETRIEVER_MODEL=intfloat/e5-base-v2

python examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path  $DATA_DIR/wiki-18/e5_Flat.index \
    --corpus_path $DATA_DIR/wiki-18/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
