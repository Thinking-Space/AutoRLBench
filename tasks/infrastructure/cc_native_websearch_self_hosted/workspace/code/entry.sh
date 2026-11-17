#!/usr/bin/env bash
set -euo pipefail

RELEASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_ROOT="$(cd -- "${RELEASE}/.." && pwd)"
RUN_ROOT="${CASE_ROOT}/runs/${JOB_ID:?JOB_ID is required}"
MODEL_DIR="${CASE_ROOT}/staged-model/Qwen3.5-9B"
SLIME_DIR="${RELEASE}/slime"
REFERENCE_DIR="${CC_SEARCH_REFERENCE_DIR:-${RUN_ROOT}/reference_torch_dist}"
export CC_SEARCH_REFERENCE_DIR="${REFERENCE_DIR}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1
export PYTHONUNBUFFERED=1
export CC_SEARCH_RUN_ROOT="${RUN_ROOT}"
export CC_BIN="${RELEASE}/bin/claude"
export CC_SEARCH_MODEL_DIR="${MODEL_DIR}"
export CC_ACTOR_MODEL=cc-search-r1-actor
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${RUN_ROOT}/wandb"
export HF_HOME="${RUN_ROOT}/hf-cache"
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${RUN_ROOT}/triton"
export no_proxy="127.0.0.1,localhost,$(hostname -i | awk '{print $1}')"
export NO_PROXY="${no_proxy}"

MEGATRON_DIR=''
for candidate in /root/Megatron-LM /opt/Megatron-LM /sgl-workspace/Megatron-LM; do
  if [[ -d "${candidate}/megatron" ]]; then
    MEGATRON_DIR="${candidate}"
    break
  fi
done
test -n "${MEGATRON_DIR}"
export PYTHONPATH="${CASE_ROOT}/dependencies:${SLIME_DIR}:${RELEASE}:${MEGATRON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"
cd "${SLIME_DIR}"

python3 - <<'PY'
import hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path
root = Path(os.environ['CC_SEARCH_RUN_ROOT'])
model = Path(os.environ['CC_SEARCH_MODEL_DIR'])
config = json.loads((model/'config.json').read_text())['text_config']
assert (config['num_hidden_layers'],config['hidden_size'],config['intermediate_size']) == (32,4096,12288)
assert (config['num_attention_heads'],config['num_key_value_heads'],config['head_dim']) == (16,4,256)
gpus = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)
assert len(gpus.strip().splitlines()) == int(os.environ.get('CC_SEARCH_LOCAL_GPUS','8'))
versions = {}
for name in ('torch','sglang','transformers','mbridge','ray','flash-linear-attention'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
record={'nodes':int(os.environ.get('CC_SEARCH_NODES','1')),'total_gpus':8,'job_id':os.environ['JOB_ID'],'gpus':gpus,'versions':versions,'python':sys.version,
        'hf_checkpoint':str(model),'model_config_sha256':hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
        'scope':'3 real GRPO rollouts; Search-R1 QA inside CC with native WebSearch; no scripted responses',
        'reward':'1.0 normalized EM; 0.2 valid answer tags but wrong answer; 0 otherwise',
        'topology':'Reference fully-async topology: 4 actor GPUs TP2/PP1/CP1/DP2 + 4 rollout GPUs (2 TP2 engines)',
        'reference_torch_dist':os.environ['CC_SEARCH_REFERENCE_DIR'],'apply_chat_template':True,
        'rollouts':3,'prompts_per_rollout':8,'samples_per_prompt':4,'global_batch_size':32}
(root/'effective-config.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
PY

# Reuse a verified initial conversion if supplied; otherwise convert the selected HF model.
if [[ -f "${REFERENCE_DIR}/latest_checkpointed_iteration.txt" ]]; then
python3 - <<'CHECK_REFERENCE'
from pathlib import Path
import hashlib,json,os
reference=Path(os.environ['CC_SEARCH_REFERENCE_DIR'])
old=json.loads((reference.parent/'effective-config.json').read_text())
model=Path(os.environ['CC_SEARCH_MODEL_DIR'])
assert old['model_config_sha256']==hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
assert (reference/'latest_checkpointed_iteration.txt').read_text().strip()=='release'
assert (reference/'release').is_dir()
print('REUSING_VERIFIED_FRESH_REFERENCE',reference,flush=True)
CHECK_REFERENCE
else
  torchrun --standalone --nproc_per_node="${CC_SEARCH_LOCAL_GPUS:-8}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --bf16 --hf-checkpoint "${MODEL_DIR}" \
    --save "${REFERENCE_DIR}" \
    --attention-backend flash --attention-dropout 0 --hidden-dropout 0 \
    2>&1 | tee "${RUN_ROOT}/conversion.log"
fi

ray start --head --node-ip-address "${CC_SEARCH_NODE_IP:-127.0.0.1}" --num-gpus "${CC_SEARCH_LOCAL_GPUS:-8}" --num-cpus 32 \
  --dashboard-host 127.0.0.1 --disable-usage-stats
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
if [[ "${CC_SEARCH_NODES:-1}" == 2 ]]; then
  python3 - <<'RAY_HEAD'
import json,os
from pathlib import Path
p=Path(os.environ['CC_SEARCH_RUN_ROOT'])/'ray-head.json'
tmp=p.with_suffix('.part');tmp.write_text(json.dumps({'ip':os.environ['CC_SEARCH_NODE_IP']}));tmp.replace(p)
RAY_HEAD
fi

python3 - <<'PY'
import json,os
from pathlib import Path
names=('PYTHONPATH','CC_SEARCH_RUN_ROOT','CC_BIN','CC_ACTOR_MODEL','CUDA_DEVICE_MAX_CONNECTIONS',
       'NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_IB_DISABLE','OMP_NUM_THREADS','HF_HUB_DISABLE_IMPLICIT_TOKEN','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
       'WANDB_MODE','WANDB_DIR','HF_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','NO_PROXY','no_proxy')
path=Path(os.environ['CC_SEARCH_RUN_ROOT'])/'ray-runtime.json'
path.write_text(json.dumps({'env_vars':{name:os.environ[name] for name in names if name in os.environ}},indent=2))
PY

ray job submit --address http://127.0.0.1:8265 \
  --runtime-env "${RUN_ROOT}/ray-runtime.json" \
  -- python3 train_async.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus 4 --num-gpus-per-node "${CC_SEARCH_LOCAL_GPUS:-8}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" --ref-load "${REFERENCE_DIR}" \
  --save "${RUN_ROOT}/checkpoints" --save-interval 3 --no-save-optim --no-save-rng \
  --prompt-data "${CASE_ROOT}/assets/search-r1-cc-train.jsonl" \
  --input-key prompt --label-key label --metadata-key metadata --apply-chat-template --rollout-shuffle \
  --num-rollout 3 --rollout-batch-size 8 --n-samples-per-prompt 4 \
  --rollout-max-response-len 2048 --rollout-max-context-len 8192 --seq-length 8192 \
  --rollout-temperature 1 --rollout-stop-token-ids 248046 248044 \
  --global-batch-size 32 --balance-data \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
  --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
  --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine 2 --sglang-server-concurrency 4 --sglang-mem-fraction-static 0.7 \
  --sglang-context-length 8192 --sglang-max-running-requests 4 --sglang-cuda-graph-max-bs 4 \
  --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
  --sglang-served-model-name cc-search-r1-actor \
  --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 --attention-backend flash \
  --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
  --custom-generate-function-path cc_search_rollout.generate \
  --custom-rm-path cc_search_rollout.reward_func \
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt" \
  --save-debug-train-data "${RUN_ROOT}/train_dumps/train_{rollout_id}.pt"

python3 "${RELEASE}/summarize.py" "${RUN_ROOT}"
