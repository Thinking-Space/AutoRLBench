#!/usr/bin/env bash
# verl_math_verify_thread_zero_reward: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 image, **the only
# networked stage**, and $AUTORLBENCH_WORKSPACE is writable. It fetches the
# model, GSM8K, MATH-lighteval and MATH-500, installs math-verify (not part
# of the verl image) into third_party/python from pinned wheels, and builds
# the parquet files exactly as the recorded runs did: verl's
# examples/data_preprocess/{gsm8k,math_dataset}.py from the workspace
# checkout, plus tools/prepare_math500.py.
#
# Destinations are what run_grpo.sh reads: MODEL=$WS/models/Qwen2.5-1.5B-Instruct,
# DATA=$WS/data/{gsm8k/train,gsm8k/test,math/train,math500/test}.parquet, PYTHONPATH+=$WS/third_party/python.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$W/.tmp" "$W/.home" "$W/third_party/wheels"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- image-level env, seeded here --------------------------------------------
# The retired `verl` recipe baked CUDA_DEVICE_ORDER/TOKENIZERS_PARALLELISM as
# image ENV; those two entries ride the prepared/env handoff into the
# agent/rerun/verifier stages via the activate hook. WANDB_MODE/WANDB_API_KEY
# are exported by the harness itself for every stage. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# The task-local fetch_hf.py fetches the whole repos, cache-first, without
# checksum verification; the revisions the former autorlbench_fetch_hf
# allowlist pinned, for reference:
#   Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
#   openai/gsm8k @ 740312add88f781978c0658806c59bc2815b9866
#   DigitalLearningGmbH/MATH-lighteval @ 0530c78699ea5e8eb5530600900e1f328b48acad
#   HuggingFaceH4/MATH-500 @ 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
#   (the MATH-lighteval fetch pulled all eight configs: `datasets` resolves
#   every config listed in the README, not only `default`.)
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo openai/gsm8k \
    --dest "$W/data/raw/gsm8k"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo DigitalLearningGmbH/MATH-lighteval \
    --dest "$W/data/raw/MATH-lighteval"
python3 "$env_dir/fetch_hf.py" \
    --type dataset \
    --repo HuggingFaceH4/MATH-500 \
    --dest "$W/data/raw/MATH-500"

# --- math-verify (used by reward/mixed_math_reward.py) -------------------------
# Pure-Python wheels from PyPI, downloaded with the pins the former
# autorlbench_fetch_url carried (sha256 + byte size verified in-flight; a
# drift on either side fails the prepare stage loudly).
python3 -B - "$W/third_party/wheels" <<'PY'
import hashlib
import sys
import urllib.request
from pathlib import Path

WHEELS = [
    (
        "https://files.pythonhosted.org/packages/62/76/6b4969bccc842b6567f7e6ee015684b9428a9b7fcbdf479e73716f43597f/math_verify-0.9.0-py3-none-any.whl",
        "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194",
        29967,
    ),
    (
        "https://files.pythonhosted.org/packages/e9/61/f75cd1fa54d8434276126034aed54dd120747de9a8fa013cdd79545ccbeb/latex2sympy2_extended-1.11.0-py3-none-any.whl",
        "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234",
        209050,
    ),
]

out_dir = Path(sys.argv[1])
for url, sha256, size in WHEELS:
    dest = out_dir / url.rsplit("/", 1)[-1]
    if dest.is_file() and dest.stat().st_size == size:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == sha256:
            print(f"[wheels] cache hit {dest.name}", flush=True)
            continue
    print(f"[wheels] downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as fh:
        got = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            fh.write(block)
            got += len(block)
            if got > size:
                raise SystemExit(f"{url}: larger than the pinned {size} bytes")
    if got != size:
        raise SystemExit(f"{url}: expected {size} bytes, got {got}")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{url}: sha256 {digest} != pinned {sha256}")
PY

if [ ! -f "$W/third_party/python/.complete" ]; then
    # Pure-Python wheels; their only other dependencies (sympy, antlr4-python3-runtime 4.9.3) are in the image.
    python3 -m pip install --no-deps --no-index --no-compile --target "$W/third_party/python" \
        "$W/third_party/wheels/math_verify-0.9.0-py3-none-any.whl" \
        "$W/third_party/wheels/latex2sympy2_extended-1.11.0-py3-none-any.whl"
    touch "$W/third_party/python/.complete"
fi

# --- Parquet files -------------------------------------------------------------
if [ ! -f "$W/data/.complete" ]; then
    export PYTHONPATH="$W/verl"
    python3 "$W/verl/examples/data_preprocess/gsm8k.py" \
        --local_dataset_path "$W/data/raw/gsm8k" --local_save_dir "$W/data/gsm8k"
    python3 "$W/verl/examples/data_preprocess/math_dataset.py" \
        --local_dataset_path "$W/data/raw/MATH-lighteval" --local_save_dir "$W/data/math"
    python3 "$W/tools/prepare_math500.py" "$W/data/raw/MATH-500" "$W/data/math500"
    # Byte-identical to the files the recorded runs used: 7473 / 1319 GSM8K, 7500 MATH train, 500 MATH-500.
    (cd "$W/data" && sha256sum -c - <<'EOF'
8fb13a0cd8621e5fcacdee3ffe28b3a369975478ea7ab514781f3f190cba2488  gsm8k/train.parquet
0dcd50ed32caa9d8f434d4e3ef8031bc3e49b0f823655a53be5c002e727f17dd  gsm8k/test.parquet
4aa303577e968be6f810600730c970b14be659a9ae9072cc2afefd04d67e7cb8  math/train.parquet
5a9c12076f5019336fc3c55449ee45c6b1e09cf1b0d1707bf9694da15d1bf4e6  math500/test.parquet
EOF
    )
    touch "$W/data/.complete"
fi
