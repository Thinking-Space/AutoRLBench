#!/bin/bash
#
# Multi-Turn PPO on Endless Terminals with Qwen3-1.7B, on slime v0.3.2.
#
# The run has two stages:
#
#   Stage A -- "the previous training run".  A real PPO rollout (actor + critic)
#              starting from the base HF model, which saves a Megatron
#              checkpoint for both roles.
#
#   Stage B -- "the model was reloaded".  The actor genuinely resumes from
#              Stage A's Megatron checkpoint (weights *and* optimizer state).
#              The critic's backbone is reloaded from the base model.  Stage B
#              opens with NUM_CRITIC_ONLY_STEPS critic-only rollouts (the
#              warmup window) and then trains actor and critic jointly.
#
# The full slime source tree is bundled in the workspace (slime/) and is the
# copy that runs -- this script puts it ahead of the image's install on
# PYTHONPATH, so editing slime/ here is what takes effect, with nothing to
# install first.
#
# NOTE ON A SHARED HOST: unlike slime's own example scripts this one never runs
# `pkill -9 ray / pkill -9 python / ray stop --force` on entry.  It is meant to
# run inside its own container, and it brings up a private Ray head on a
# non-default port with a private temp dir.

set -euo pipefail

WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# The container runs as root while /workspace is a bind mount owned by whoever
# checked the case out, so any .pyc written here lands root-owned in their tree
# -- and `importlib` copies the *source* file's mode onto the .pyc, so a 600
# source yields a 600 root-owned file that even the owner cannot read, and
# `cp -r` of the case fails. Nothing here benefits from a bytecode cache.
export PYTHONDONTWRITEBYTECODE=1

MODEL_DIR="${MODEL_DIR:-${WS}/Qwen3-1.7B}"
ET_SRC="${ET_SRC:-${WS}/endless-terminals}"
SLIME_HOME="${SLIME_HOME:-${WS}/slime}"
OUT_DIR="${OUT_DIR:-${WS}/outputs}"
CKPT_ROOT="${CKPT_ROOT:-/ckpt}"

# Endless Terminals sandbox knobs; also forwarded into the Ray workers, which
# read them at import time in et/rollout.py.
#
# These must NOT live under /dev/shm: it is mounted noexec in the training
# image, and many tasks' verifiers assert os.access(path, os.X_OK), which is
# False there even at mode 755.
export ET_DATA_ROOT="${ET_DATA_ROOT:-/tmp/et_prompts}"
export ET_SCRATCH="${ET_SCRATCH:-/tmp/et_episodes}"
export ET_MAX_TURNS="${ET_MAX_TURNS:-6}"
export ET_MAX_NEW_TOKENS="${ET_MAX_NEW_TOKENS:-384}"
export ET_CMD_TIMEOUT="${ET_CMD_TIMEOUT:-20}"
export ET_MAX_OBS_CHARS="${ET_MAX_OBS_CHARS:-1200}"

NUM_GPUS="${NUM_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.55}"
# Rollouts per stage. Both stages use the same count on purpose, and must:
# slime derives `train_iters` from `--num-rollout` (model.py:203) and Megatron
# stores the resulting LR/weight-decay horizons inside the checkpoint, then
# asserts on reload that they match the ones the new arguments imply
# ("total number of iterations do not match"). Giving Stage B a different
# rollout count than Stage A would trip that assert before any training runs.
# In Stage A these rollouts are ordinary PPO; Stage B opens with
# NUM_CRITIC_ONLY_STEPS critic-only rollouts -- the warmup window -- and then
# trains actor and critic jointly for the rest.
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
# Stage B is the long one: critic warmup followed by joint PPO, trained until
# the validation pass rate stops moving. It no longer has to match Stage A's
# rollout count -- see --override-opt-param-scheduler at the submit site.
NUM_ROLLOUTS_B="${NUM_ROLLOUTS_B:-${NUM_ROLLOUTS}}"
# Rollouts at the start of Stage B in which only the critic trains. This is the
# warmup window the incident was reported against, and the window in which the
# value loss is measured.
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-${NUM_ROLLOUTS_B}}"

# Episodes per rollout = ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT; optimizer
# steps per rollout = that, divided by GLOBAL_BATCH_SIZE.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LR="${LR:-1e-6}"

# How many leading critic optimizer steps the reported warmup statistics cover.
# Derived rather than guessed, because once Stage B runs past the critic-only
# phase the two are no longer the same thing: averaging the value loss over all
# of a 30-rollout Stage B measures mostly a critic that has already been
# trained, which is not what "warmup value loss" names and not what the
# incident was about.
CRITIC_WARMUP_STEPS=$(( NUM_CRITIC_ONLY_STEPS * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / GLOBAL_BATCH_SIZE ))
[ "${CRITIC_WARMUP_STEPS}" -gt 0 ] || CRITIC_WARMUP_STEPS=1

# Held-out evaluation during Stage B. 0 disables it, which is what the short
# value-loss-only reproduction wants.
#
# val and test get different sample budgets. val only has to say *when* the
# curve flattens, so 2 samples x 150 tasks is enough; test carries the
# end-to-end claim and is compared per task, where the useful quantity is each
# task's success *rate*, which 1 or 2 samples cannot resolve at all.
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-2}"
N_EVAL_SAMPLES_TEST="${N_EVAL_SAMPLES_TEST:-4}"

# How often Stage B checkpoints. Trained to convergence, Stage B runs for well
# over an hour, which on a shared host is long enough to be interrupted -- two
# runs of this case were killed from outside the container mid-training, once
# with SIGTERM and once with SIGKILL. So Stage B saves as it goes and a second
# invocation picks up where the first stopped, instead of starting the hour
# again. 0 disables it, which is what the short value-loss-only reproduction
# wants: it writes nothing and measures only the reload.
STAGE_B_SAVE_INTERVAL="${STAGE_B_SAVE_INTERVAL:-0}"

RAY_PORT="${RAY_PORT:-6479}"
DASH_PORT="${DASH_PORT:-8277}"
RAY_TMP="${RAY_TMP:-/tmp/ray_ppo}"

mkdir -p "${OUT_DIR}" "${CKPT_ROOT}" "${ET_SCRATCH}"

# --------------------------------------------------------- 0. the slime source
# The full slime source tree is bundled in the workspace (${SLIME_HOME}) and is
# put ahead of the image's editable install on PYTHONPATH below, so it is the
# copy that trains -- there is nothing to install first.
test -f "${SLIME_HOME}/train.py" || {
    echo "FATAL: bundled slime source not found at ${SLIME_HOME}" >&2
    exit 1
}

# ------------------------------------------------------- 1. the prompt dataset
# prepare.sh ships the already-built dataset (transpiled + validated + split) at
# ${WS}/et_prompts. Seed ET_DATA_ROOT from it so the build below is skipped: the
# 2492->1489 admission is environment-fixed and reproducible, so there is no need
# to re-run the ~40min transpile+validate pass (and ET_SRC is never read). The
# copy lands in a writable, exec-capable dir (ET_DATA_ROOT defaults to /tmp) --
# the bundled tree under /workspace may be a root-owned bind mount. If the
# prebuilt payload is absent, fall through to building from ET_SRC as before.
if [ ! -s "${ET_DATA_ROOT}/endless_terminals.jsonl" ] && [ -s "${WS}/et_prompts/endless_terminals.jsonl" ]; then
    echo "=== seeding ET_DATA_ROOT from prebuilt ${WS}/et_prompts (skipping transpile+validate) ==="
    mkdir -p "${ET_DATA_ROOT}"
    cp -a "${WS}/et_prompts/." "${ET_DATA_ROOT}/"
fi
if [ ! -s "${ET_DATA_ROOT}/endless_terminals.jsonl" ]; then
    echo "=== transpiling + validating the Endless Terminals tasks ==="
    (cd "${WS}" && python3 -m et.build_dataset --src "${ET_SRC}" --out "${ET_DATA_ROOT}") \
        2>&1 | tee "${OUT_DIR}/build_dataset.log"
fi
# Deliberately not silenced. A report that is quietly 0 bytes is worse than one
# that is loudly absent: it reads as "every task was rejected" to anyone who
# opens it later, and that has happened here more than once.
copy_artifact() {   # copy_artifact <src>
    local src="$1" dst="${OUT_DIR}/$(basename "$1")"
    if ! cp -f "${src}" "${dst}"; then
        echo "WARNING: could not copy ${src} into ${OUT_DIR}" >&2
    elif [ ! -s "${dst}" ] && [ -s "${src}" ]; then
        echo "WARNING: ${dst} came out empty although ${src} is not" >&2
    fi
}
copy_artifact "${ET_DATA_ROOT}/validation_report.json"
N_PROMPTS="$(wc -l < "${ET_DATA_ROOT}/endless_terminals.jsonl")"
echo "prompt set: ${N_PROMPTS} validated Endless Terminals tasks"
if [ "${N_PROMPTS}" -lt 4 ]; then
    echo "FATAL: too few tasks survived validation to train on." >&2
    exit 1
fi

# Train on one part of the task set and measure on another. The split is a pure
# function of the admitted list and a fixed seed, so every run trains and is
# scored on exactly the same tasks -- otherwise a comparison between runs is
# partly measuring which tasks each one happened to get.
if [ ! -s "${ET_DATA_ROOT}/endless_terminals_train.jsonl" ]; then
    (cd "${WS}" && python3 -m et.make_splits --prompts "${ET_DATA_ROOT}/endless_terminals.jsonl")
fi
copy_artifact "${ET_DATA_ROOT}/splits.json"
TRAIN_SPLIT="${ET_DATA_ROOT}/endless_terminals_train.jsonl"
VAL_SPLIT="${ET_DATA_ROOT}/endless_terminals_val.jsonl"
TEST_SPLIT="${ET_DATA_ROOT}/endless_terminals_test.jsonl"
printf 'splits: train %s / val %s / test %s\n' \
    "$(wc -l < "${TRAIN_SPLIT}")" "$(wc -l < "${VAL_SPLIT}")" "$(wc -l < "${TEST_SPLIT}")"

# ------------------------------------------------------------- 2. shared args
source "${WS}/configs/model_args.sh"

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_SPLIT}"
    --input-key prompt
    --label-key label
    --rollout-shuffle
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 2048
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    # The multi-turn terminal loop and its pytest-based episode reward.
    --custom-generate-function-path et.rollout.generate
    --custom-rm-path et.rollout.reward_func
)

PPO_ARGS=(
    --advantage-estimator ppo   # -> args.use_critic
    --gamma 1.0
    --lambd 1.0
    --eps-clip 0.2
    --value-clip 0.2
    --kl-coef 0.0               # no reference model, hence no --ref-load and
    --entropy-coef 0.0          # no HF -> torch_dist conversion step
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)
# Sequence parallelism splits the activations of the non-tensor-parallel ops
# across the TP group, so it is only defined for TP > 1; Megatron asserts on it.
# Spelled as an `if` rather than `[ ... ] && ...`: the latter evaluates to the
# test's status, which under `set -e` ends the script whenever TP_SIZE is 1.
if [ "${TP_SIZE}" -gt 1 ]; then
    PERF_ARGS+=(--sequence-parallel)
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${NUM_GPUS}"
    # Colocated: this fraction is what SGLang keeps, and Megatron's weights,
    # gradients and Adam state have to fit in the rest. At TP 2 each GPU carries
    # half of each role, so 0.55 leaves enough; drop it when NUM_GPUS is 1 and
    # one device has to hold both roles whole.
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)

# Held-out scoring, used only by Stage B. slime evaluates once before the first
# rollout and then every EVAL_INTERVAL rollouts, which gives both the starting
# pass rate and the curve that says whether training has converged.
#
# val and test are evaluated at the same points on purpose. The convergence
# *decision* is read off val; test is reported only at the final point, and
# evaluating it throughout costs generation but no statistical validity, since
# nothing about the run is conditioned on it.
#
# Spelled as an --eval-config YAML rather than --eval-prompt-data because the
# two splits need different sample counts, and --n-samples-per-eval-prompt is
# one global number. Everything the YAML leaves out still falls back to the
# command line (eval_config.py:10 DATASET_RUNTIME_SPECS), so temperature,
# response length and the prompt/label keys are set once, below and above.
EVAL_ARGS=()
if [ "${EVAL_INTERVAL}" -gt 0 ]; then
    cat > "${OUT_DIR}/eval_datasets.yaml" <<YAML
eval:
  datasets:
    - name: et_val
      path: ${VAL_SPLIT}
      n_samples_per_eval_prompt: ${N_EVAL_SAMPLES}
    - name: et_test
      path: ${TEST_SPLIT}
      n_samples_per_eval_prompt: ${N_EVAL_SAMPLES_TEST}
YAML
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-config "${OUT_DIR}/eval_datasets.yaml"
        --eval-max-response-len 2048
        --eval-temperature 1.0
        --log-passrate
        --custom-eval-rollout-log-function-path et.eval_log.log_eval_rollout_data
    )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed 1234
)

# --------------------------------------------------------- 3. per-role configs
# Actor and critic share `args.save` by default, which would make them clobber
# each other's checkpoints (the critic's output layer is [1, hidden_size], the
# actor's is [vocab, hidden_size]), so each role gets its own directory.
cat > "${OUT_DIR}/megatron_roles_stage_a.yaml" <<YAML
# Stage A: "the previous training run". Both roles start from the base model.
megatron:
  - name: stage-a-actor
    role: actor
    overrides:
      load: ${MODEL_DIR}
      save: ${CKPT_ROOT}/actor
  - name: stage-a-critic
    role: critic
    overrides:
      load: ${MODEL_DIR}
      save: ${CKPT_ROOT}/critic
YAML

# Stage B can be entered two ways, and they are not the same event.
#
#   fresh   -- "the model was reloaded". The actor resumes from Stage A's
#              Megatron checkpoint; the critic's backbone comes from the base
#              model.
#
#   resume  -- an interrupted Stage B being continued. Both roles load Stage B's
#              own checkpoints, optimizer state included, and the run continues
#              from the last saved rollout. The first attempt's log is kept,
#              not overwritten (see the STAGE_B_LOG choice below).
STAGE_B_RESUME=0
if [ -f "${CKPT_ROOT}/stage_b_actor/latest_checkpointed_iteration.txt" ] \
   && [ -f "${CKPT_ROOT}/stage_b_critic/latest_checkpointed_iteration.txt" ]; then
    STAGE_B_RESUME=1
fi

if [ "${STAGE_B_RESUME}" = "1" ]; then
    STAGE_B_ACTOR_LOAD="${CKPT_ROOT}/stage_b_actor"
    STAGE_B_CRITIC_LOAD="${CKPT_ROOT}/stage_b_critic"
else
    STAGE_B_ACTOR_LOAD="${CKPT_ROOT}/actor"
    STAGE_B_CRITIC_LOAD="${MODEL_DIR}"
fi

cat > "${OUT_DIR}/megatron_roles_stage_b.yaml" <<YAML
# Stage B, $([ "${STAGE_B_RESUME}" = "1" ] && echo "resumed from its own checkpoints" || echo "entered fresh after a reload").
megatron:
  - name: stage-b-actor
    role: actor
    overrides:
      load: ${STAGE_B_ACTOR_LOAD}
      save: ${CKPT_ROOT}/stage_b_actor
  - name: stage-b-critic
    role: critic
    overrides:
      load: ${STAGE_B_CRITIC_LOAD}
      save: ${CKPT_ROOT}/stage_b_critic
YAML

# ------------------------------------------------------------------- 4. ray
export PYTHONUNBUFFERED=1

ray start --head \
    --node-ip-address 127.0.0.1 \
    --port "${RAY_PORT}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 127.0.0.1 \
    --dashboard-port "${DASH_PORT}" \
    --temp-dir "${RAY_TMP}"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

# `|| true` on the whole pipeline, not decoration: under `pipefail` a `grep`
# that matches nothing fails the pipeline, so on a host with no NVLink -- or in
# a container holding a single GPU, where the topology matrix has no peer links
# to report -- `set -e` would kill the script here. Silently: this sits between
# `ray start` and the first echo of Stage A, so the log just stops.
NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
HAS_NVLINK=0
[ "${NVLINK_COUNT:-0}" -gt 0 ] && HAS_NVLINK=1

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_HOME}:${WS}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"ET_DATA_ROOT\": \"${ET_DATA_ROOT}\",
    \"ET_EVAL_DUMP_DIR\": \"${OUT_DIR}/eval_episodes\",
    \"ET_SCRATCH\": \"${ET_SCRATCH}\",
    \"ET_MAX_TURNS\": \"${ET_MAX_TURNS}\",
    \"ET_MAX_NEW_TOKENS\": \"${ET_MAX_NEW_TOKENS}\",
    \"ET_CMD_TIMEOUT\": \"${ET_CMD_TIMEOUT}\",
    \"ET_MAX_OBS_CHARS\": \"${ET_MAX_OBS_CHARS}\"
  }
}"

submit() {   # submit <log-file> <extra args...>
    local log="$1"; shift
    # The entrypoint is an absolute path on purpose: `ray job submit` does not
    # run it in this shell's directory but in the cwd of the Ray head (here
    # /workspace, since no `working_dir` is uploaded), so a bare `train.py`
    # would not be found. Running ${SLIME_HOME}/train.py puts the bundled slime
    # tree on the driver's sys.path[0]; the ray workers get it from ${SLIME_HOME}
    # on PYTHONPATH (RUNTIME_ENV_JSON above), ahead of the image's install.
    ray job submit \
        --address "http://127.0.0.1:${DASH_PORT}" \
        --runtime-env-json "${RUNTIME_ENV_JSON}" \
        -- python3 "${SLIME_HOME}/train.py" \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node "${NUM_GPUS}" \
        --colocate \
        --hf-checkpoint "${MODEL_DIR}" \
        "${MODEL_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${PPO_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}" \
        "$@" 2>&1 | tee "${log}"
}

# Ray forwards actor stdout to the job driver, but keep the raw worker logs too:
# the per-step metrics are emitted by the train actors, not by the driver.
collect_worker_logs() {   # collect_worker_logs <tag>
    # Two statements, not one: `local a=.. b=..` expands every word before it
    # assigns any of them, so referring to ${tag} on the same line would read
    # it while it is still unset -- fatal under `set -u`.
    local tag="$1"
    local dst="${OUT_DIR}/${tag}_worker_logs"
    mkdir -p "${dst}"
    grep -rl "critic-step" "${RAY_TMP}"/session_*/logs/ 2>/dev/null | while read -r f; do
        cp -f "${f}" "${dst}/$(basename "$(dirname "${f}")")_$(basename "${f}")" 2>/dev/null || true
    done
}

# ---------------------------------------------------------------- 5. stage A
echo
echo "############ Stage A: the previous training run ############"
# SKIP_STAGE_A=1 reuses a checkpoint an earlier invocation already wrote, for
# re-running the long Stage B after an interruption. Off by default, and
# refuses to be silently wrong: with no checkpoint there, Stage A runs anyway.
if [ "${SKIP_STAGE_A:-0}" = "1" ] && [ -f "${CKPT_ROOT}/actor/latest_checkpointed_iteration.txt" ]; then
    echo "SKIP_STAGE_A=1 and ${CKPT_ROOT}/actor already holds a checkpoint -- reusing it."
else
    submit "${OUT_DIR}/stage_a.log" \
        --megatron-config-path "${OUT_DIR}/megatron_roles_stage_a.yaml" \
        --load "${MODEL_DIR}" \
        --save "${CKPT_ROOT}/actor" \
        --save-interval 1 \
        --num-rollout "${NUM_ROLLOUTS}" \
        --num-critic-only-steps 0
    collect_worker_logs stage_a
fi

echo
echo "checkpoint written by Stage A:"
ls -1 "${CKPT_ROOT}/actor" 2>/dev/null || true
test -f "${CKPT_ROOT}/actor/latest_checkpointed_iteration.txt" || {
    echo "FATAL: Stage A did not produce a resumable Megatron checkpoint." >&2
    exit 1
}

# ---------------------------------------------------------------- 6. stage B
echo
echo "############ Stage B: reload, then critic warmup ############"
# --save-interval has to be set because --save is: Megatron's validate_args
# asserts the two go together. With STAGE_B_SAVE_INTERVAL=0 it is set far
# larger than the number of rollouts Stage B takes, so Stage B measures the
# reload and writes nothing -- the only checkpoint the short reproduction needs
# is the one Stage A already produced.
#
# --override-opt-param-scheduler is what lets Stage B be longer than Stage A.
# slime derives `train_iters` from --num-rollout (model.py:203) and Megatron
# stores the resulting LR/weight-decay horizons in the checkpoint, then asserts
# on reload that the new arguments imply the same ones ("total number of
# iterations do not match"). The flag says "use the command line, not the
# checkpoint", which is exactly right here: Stage B is a new training run that
# happens to start from Stage A's weights, not a resumption of Stage A.
STAGE_B_ARGS=(
    --megatron-config-path "${OUT_DIR}/megatron_roles_stage_b.yaml"
    --load "${STAGE_B_ACTOR_LOAD}"
    --save "${CKPT_ROOT}/stage_b_actor"
    --save-interval "$([ "${STAGE_B_SAVE_INTERVAL}" -gt 0 ] && echo "${STAGE_B_SAVE_INTERVAL}" || echo 1000)"
    --override-opt-param-scheduler
    --num-rollout "${NUM_ROLLOUTS_B}"
    --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
)

STAGE_B_LOG="${OUT_DIR}/stage_b.log"
if [ "${STAGE_B_RESUME}" = "1" ]; then
    # The first attempt's log holds the warmup window that the value-loss
    # verdict is read from, so a resume must not overwrite it.
    n=1
    while [ -e "${OUT_DIR}/stage_b_resume_${n}.log" ]; do n=$((n + 1)); done
    STAGE_B_LOG="${OUT_DIR}/stage_b_resume_${n}.log"
    echo "resuming Stage B from ${CKPT_ROOT}/stage_b_{actor,critic}" \
         "(iteration $(cat "${CKPT_ROOT}/stage_b_critic/latest_checkpointed_iteration.txt")); log -> ${STAGE_B_LOG}"
    # --start-rollout-id is deliberately NOT passed here. slime takes the start
    # id from the *critic*'s loaded checkpoint when a critic is in play
    # (placement_group.py:212), and Stage B saves at `iteration = rollout_id`,
    # so omitting it resumes at exactly the rollout after the last saved one.
else
    # --start-rollout-id 0 pins the warmup window: without it the start id is
    # inferred from whichever checkpoint the critic loaded, and on a fresh
    # Stage B the critic loads a plain HF directory (iteration 0).
    STAGE_B_ARGS+=(--start-rollout-id 0)
fi

submit "${STAGE_B_LOG}" "${STAGE_B_ARGS[@]}" "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"
collect_worker_logs stage_b

# ---------------------------------------------------------------- 7. verdict
echo
echo "############ critic warmup value loss ############"
python3 "${WS}/tools/parse_value_loss.py" \
    "${OUT_DIR}/stage_b.log" \
    --warmup-steps "${CRITIC_WARMUP_STEPS}" \
    --out "${OUT_DIR}/stage_b_value_loss.json"

python3 "${WS}/tools/parse_value_loss.py" \
    "${OUT_DIR}/stage_a.log" \
    --out "${OUT_DIR}/stage_a_value_loss.json" >/dev/null 2>&1 || true

echo
echo "artifacts in ${OUT_DIR}:"
ls -1 "${OUT_DIR}"
