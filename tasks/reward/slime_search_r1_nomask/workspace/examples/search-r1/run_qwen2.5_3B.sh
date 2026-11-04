#!/bin/bash

# Kill leftover python from a previous run, but spare the retrieval server so it
# can be reused across reruns (reloading its ~61GB index takes minutes).
kill_python_except_retriever() {
   local keep pid
   keep=" $(pgrep -f 'local_dense_retriever/retrieval_server.py' 2>/dev/null | tr '\n' ' ')"
   for pid in $(pgrep -f 'python' 2>/dev/null); do
      case "$keep" in
         *" $pid "*) ;;                 # this is the retriever -> keep it
         *) kill -9 "$pid" 2>/dev/null ;;
      esac
   done
}

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
kill_python_except_retriever
sleep 3
pkill -9 ray
kill_python_except_retriever

set -ex

# The rollout runs up to search_concurrency (256) concurrent aiohttp requests to
# the local retrieval server, on top of sglang/ray file descriptors. The default
# soft limit (1024) is far too low and the rollout crashes with
# "OSError: [Errno 24] Too many open files". Raise it as high as the hard limit
# allows; ray workers launched below inherit this shell's limit.
ulimit -n "$(ulimit -Hn)"

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Online W&B initialization is unreliable from this devspace and a timeout
# kills the RolloutManager actor. Default to local logging; callers can opt in
# with WANDB_MODE=online once network access is stable.
export WANDB_MODE=${WANDB_MODE:-offline}

# The persistent faiss retriever (--faiss_gpu) holds ~9GB more on GPU0 than on the
# other GPUs, so SGLang's TP memory-balance check ("memory capacity is unbalanced")
# fatally aborts the GPU0-paired engine on restart. The imbalance is expected and
# within the mem-fraction-static headroom. NOTE: slime gives each rollout engine an
# isolated ray runtime_env, so exporting the var here does NOT reach those actors --
# the actual disable lives in slime/ray/rollout.py's engine env_vars. This export
# only covers the driver/train side and is kept as defense-in-depth.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# On the cybertron devspaces an egress whitelist proxy is exported into every
# shell (http_proxy/https_proxy -> whitelist-proxy:7891) with a no_proxy that
# does NOT cover localhost or this node. ray's "ray job submit" then routes its
# request to the local dashboard (127.0.0.1:8265) through that proxy and gets a
# 502 Bad Gateway, and the same happens for sglang's node-local HTTP. Prepend
# the loopback addresses, this node's IP and its /16 to no_proxy so all
# in-cluster / on-node traffic bypasses the proxy. Harmless off-cluster (the
# vars are simply absent). hostname -I is the reliable source for the node IP;
# `ip` is not always on PATH in these non-login shells, so it is only a fallback.
NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || true
if [ -z "${NODE_IP:-}" ]; then
   NODE_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}') || true
fi
case "${NODE_IP:-}" in
   *.*.*.*)
      PREFIX16=$(echo "$NODE_IP" | awk -F. '{print $1"."$2".0.0/16"}')
      export no_proxy="127.0.0.1,localhost,0.0.0.0,${NODE_IP},${PREFIX16}${no_proxy:+,$no_proxy}"
      export NO_PROXY="$no_proxy" ;;
   *)
      export no_proxy="127.0.0.1,localhost,0.0.0.0${no_proxy:+,$no_proxy}"
      export NO_PROXY="$no_proxy" ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# slime repo root (this recipe lives at <slime>/examples/search-r1)
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

# Use the HuggingFace mirror for all downloads.
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

DATA_DIR=${DATA_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$PWD}
bash "$SCRIPT_DIR/prepare_data.sh"

DATASET_DIR=${DATASET_DIR:-$DATA_DIR/nq_hotpotqa_train}
TRAIN_DATA=${TRAIN_DATA:-$DATASET_DIR/train.parquet}
TEST_DATA=${TEST_DATA:-$DATASET_DIR/test.parquet}

# Model checkpoints under MODEL_DIR: the HF checkpoint and its mcore
# (torch_dist) conversion used as the reference weights.
HF_MODEL_DIR=${HF_MODEL_DIR:-$MODEL_DIR/Qwen2.5-3B}
DIST_MODEL_DIR=${DIST_MODEL_DIR:-$MODEL_DIR/Qwen2.5-3B_torch_dist}

# Download the HF checkpoint if it is not already present under MODEL_DIR.
if [ ! -d "$HF_MODEL_DIR" ] || [ -z "$(ls -A "$HF_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] $HF_MODEL_DIR missing; downloading Qwen/Qwen2.5-3B from $HF_ENDPOINT"
   hf download Qwen/Qwen2.5-3B --local-dir "$HF_MODEL_DIR"
fi

# Convert to a torch_dist (mcore) checkpoint if not already done.
if [ ! -d "$DIST_MODEL_DIR" ] || [ -z "$(ls -A "$DIST_MODEL_DIR" 2>/dev/null)" ]; then
   echo "[model] converting $HF_MODEL_DIR -> $DIST_MODEL_DIR (torch_dist)"
   PYTHONPATH=${MEGATRON_LM_DIR:-/root/Megatron-LM} python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$HF_MODEL_DIR" \
      --save "$DIST_MODEL_DIR"
fi

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --ref-load "$DIST_MODEL_DIR"
   # --load "$MODEL_DIR/Qwen2.5-3B_slime/"
   --save "$MODEL_DIR/Qwen2.5-3B_slime/"
   --save-interval 10
)

# --- the local dense retrieval server ---------------------------------------
# generate_with_search.py posts every search turn to
# http://127.0.0.1:8000/retrieve, so the rollout cannot start without it.
#
# Started here rather than left to the caller, because none of the three paths
# that run this script are able to start it themselves:
#
#   * the harness's prepare stage is a container created and removed around
#     prepare.sh (evaluation/evaluator.py::_run_prepare, and
#     runtimes/providers/docker.py::close_sandbox is a `docker rm -f`), so a
#     server backgrounded there does not outlive the stage -- and prepare is
#     waited on in the foreground, so starting it there without detaching just
#     hangs until the stage timeout;
#   * on the harbor path prepare can be a build-time RUN layer, which has no
#     GPU at all;
#   * the solution-level instruction forbids the agent from using `&`, nohup,
#     tmux, screen or any other detach mechanism.
#
# A child of this script is none of those things: the script itself still runs
# in the foreground and still owns the exit status. It also has to happen before
# `ray start` below, so that sglang sizes its engines against the memory the
# faiss shards already hold (see the SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
# note above).
#
# An already-listening server is reused rather than restarted: loading the 60GiB
# index takes minutes, which is the same reason kill_python_except_retriever at
# the top of this file spares it across reruns.
RETRIEVAL_PORT=${RETRIEVAL_PORT:-8000}
RETRIEVAL_LOG=${RETRIEVAL_LOG:-$SLIME_DIR/retrieval_server.log}
# 30 minutes. A *cold* start also has to turn the 14GB corpus into datasets'
# arrow cache, which is the slow half; environment/prepare.sh warms that cache
# so the usual start is a few minutes. The guard has to cover the cold case
# anyway -- nothing guarantees prepare ran.
RETRIEVAL_STARTUP_TIMEOUT=${RETRIEVAL_STARTUP_TIMEOUT:-1800}

# The port opens only after retrieval_server.py has built the retriever (the
# index load and the corpus load both happen before uvicorn.run), so "listening"
# really does mean "ready" -- no separate health request needed.
retrieval_listening() { (exec 3<>"/dev/tcp/127.0.0.1/${RETRIEVAL_PORT}") 2>/dev/null; }

# Matches the start script too, not just the server: for the first few seconds
# after launch the process is still the `bash start_retriver.sh` wrapper and has
# not exec'd python yet, and treating that as "already dead" would abort a start
# that is fine.
retrieval_alive() {
   pgrep -f 'local_dense_retriever/retrieval_server.py' >/dev/null 2>&1 \
      || pgrep -f 'search-r1/start_retriver.sh' >/dev/null 2>&1
}

if retrieval_listening; then
   echo "[retrieval] reusing the server already listening on :${RETRIEVAL_PORT}"
else
   echo "[retrieval] starting the local dense retriever (log: $RETRIEVAL_LOG)"
   # Note the spelling: the file is start_retriver.sh, without the second 'e'.
   # It also carries no shebang, so it has to be handed to bash explicitly.
   #
   # DATA_DIR / MODEL_DIR are passed through rather than left to default to
   # $PWD: the server must read the same wiki-18 and e5-base-v2 this run was
   # told about, even when the caller's working directory is not $SLIME_DIR.
   #
   # setsid puts it in its own process group. A *rerun* of this script starts by
   # walking process groups with pkill/ray stop, and keeping the retriever out of
   # this shell's group is what makes kill_python_except_retriever's exemption
   # actually hold.
   #
   # The redirections sit on the *subshell*, not on the setsid command inside it.
   # This is not cosmetic. With `... setsid bash ... >"$LOG" 2>&1 &` the
   # backgrounded AND-list runs in an intermediate subshell that still holds this
   # script's own stdout and stderr, and that subshell outlives the script -- so
   # the retriever, which is meant to survive for the *next* rerun, keeps the
   # write end of this script's stdout pipe open. Whoever is reading that pipe
   # (`| tee .../train.log` under docker/debug-env.sh, the harness's own capture
   # in the agent stage) then never sees EOF, and a finished training run looks
   # like a hung one. Redirecting the subshell leaves no descendant holding it.
   (
      cd "$SLIME_DIR" \
         && DATA_DIR="$DATA_DIR" MODEL_DIR="$MODEL_DIR" \
            setsid bash examples/search-r1/start_retriver.sh &
   ) >"$RETRIEVAL_LOG" 2>&1 </dev/null

   # The wait loop is untraced on purpose: at one probe every 5s a cold start
   # would otherwise put thousands of `set -x` lines into the graded train log.
   # Progress is reported once a minute instead.
   set +x
   retrieval_waited=0
   until retrieval_listening; do
      if ! retrieval_alive; then
         echo "[retrieval] the server exited before it started listening." >&2
         echo "[retrieval] last 40 lines of $RETRIEVAL_LOG:" >&2
         tail -n 40 "$RETRIEVAL_LOG" >&2 || true
         exit 1
      fi
      if [ "$retrieval_waited" -ge "$RETRIEVAL_STARTUP_TIMEOUT" ]; then
         echo "[retrieval] still no listener on :${RETRIEVAL_PORT} after ${RETRIEVAL_STARTUP_TIMEOUT}s." >&2
         echo "[retrieval] last 40 lines of $RETRIEVAL_LOG:" >&2
         tail -n 40 "$RETRIEVAL_LOG" >&2 || true
         exit 1
      fi
      sleep 5
      retrieval_waited=$((retrieval_waited + 5))
      if [ $((retrieval_waited % 60)) -eq 0 ]; then
         echo "[retrieval] loading the index; ${retrieval_waited}s elapsed"
      fi
   done
   set -x
fi
echo "[retrieval] server up on :${RETRIEVAL_PORT}"

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-context-len 8129
   --rollout-max-response-len 1024
   --rollout-temperature 1

   # eval args
   --eval-interval 10
   --eval-prompt-data nq_test "$TEST_DATA@[0:3000]"
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt 1

   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-search-r1
   --wandb-group search-r1_qwen2.5-3B
   # wandb runs offline here, so no real key is ever used; --wandb-key still
   # needs *an* argument or argparse aborts with "expected one argument" when
   # WANDB_API_KEY is unset. The literal EMPTY is the placeholder slime accepts.
   --wandb-key "${WANDB_API_KEY:-EMPTY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR:-/root/Megatron-LM}/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
