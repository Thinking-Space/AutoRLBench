#!/usr/bin/env bash
# verl_resume_restores_checkpoint_lr: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 image, **the only networked stage**, and
# $AUTORLBENCH_WORKSPACE is writable. It fetches the start model and installs
# inspect_ai (used by scripts/math_reward.py; not part of the image) plus the
# packages it imports that the image lacks into third_party/python, from pinned
# wheels. verl v0.8.0 (commit 7aed6b230776f963fa09509c10d9c3a767d1102c) ships
# in workspace/verl; the training parquet is built at launch by run_grpo.sh from
# workspace/data/pool_clean.jsonl (run_resume.sh calls run_grpo.sh once per stage).
#
# Destinations are what run_grpo.sh reads: MODEL=$WS/models/Qwen3-4B,
# PYTHONPATH=$WS/verl:$WS/third_party/python.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
env_dir="$AUTORLBENCH_ENVIRONMENT"
mkdir -p "$W/.tmp" "$W/.home" "$W/third_party/wheels"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

for _kv in TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- Base model ---------------------------------------------------------------
# run_grpo.sh defaults MODEL to $WS/models/Qwen3-4B.
# Pinned revision, for reference (fetch_hf.py takes no revision argument): Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
python3 "$env_dir/fetch_hf.py" \
    --repo Qwen/Qwen3-4B \
    --dest "$W/models/Qwen3-4B"

# --- inspect_ai and the modules it imports that the image lacks ------------------
# aiobotocore 3.10.0, aioitertools 0.13.0, boto3 1.44.7, botocore 1.44.7, frozendict 2.5.1, inspect_ai 0.3.150, jsonlines 4.0.0, jsonpatch 1.48, jsonpointer 3.12, mmh3 5.3.1, nest_asyncio 1.6.0, s3fs 2026.9.0, s3transfer 0.19.3, semver 3.1.2, shortuuid 1.0.13, tenacity 9.2.1, textual 8.2.8
# Wheels from PyPI; sha256 + byte size verified in-flight, a drift fails loudly.
python3 -B - "$W/third_party/wheels" <<'PY'
import hashlib
import sys
import urllib.request
from pathlib import Path

WHEELS = [
    (
        "https://files.pythonhosted.org/packages/7f/bc/3bb8965204ba51bbf4d021bc89d1357cd2179a17ef71b5fc429e21b511a2/aiobotocore-3.10.0-py3-none-any.whl",
        "de0ccbcb82b6b0009f1b6667f48442e9d0432f9b8bfe6172fa51777cbea89d03",
        102759,
    ),
    (
        "https://files.pythonhosted.org/packages/10/a1/510b0a7fadc6f43a6ce50152e69dbd86415240835868bb0bd9b5b88b1e06/aioitertools-0.13.0-py3-none-any.whl",
        "0be0292b856f08dfac90e31f4739432f4cb6d7520ab9eb73e143f4f2fa5259be",
        24182,
    ),
    (
        "https://files.pythonhosted.org/packages/55/39/1b35b73d9f62c9a722ece0eed6673de7a56803bfb1f89926a36445cceab9/boto3-1.44.7-py3-none-any.whl",
        "15fac273416d37cd5e3f0c02b33fd8d7f27d3e64c5f64b911f8b953772302609",
        140157,
    ),
    (
        "https://files.pythonhosted.org/packages/0b/47/a6d5d012e4d6e70f80d4d0dffac7622f8f64edce635200ea23a3c531f397/botocore-1.44.7-py3-none-any.whl",
        "56ec7cf3a7c0012c7e105c8f498c67077afa722df916dbdbe667b584fe7e5f99",
        16741609,
    ),
    (
        "https://files.pythonhosted.org/packages/60/98/0902e074fb777b4064594a3a18773a50d661dbb8d73b7487fdf63ba088cc/frozendict-2.5.1-py3-none-any.whl",
        "b6e602e286b912beca1e12615f50eda6c74646b440c8cfde34221adee6c98c5e",
        18055,
    ),
    (
        "https://files.pythonhosted.org/packages/16/b0/af64505c8161d57df50a851462b9241b51b63a160ff1cd4f1b4368de2f79/inspect_ai-0.3.150-py3-none-any.whl",
        "b90a687ee4adb0b322bdb2c1f751ba09c621527ba8f8b7c49aeff12f7ae7434b",
        34636108,
    ),
    (
        "https://files.pythonhosted.org/packages/f8/62/d9ba6323b9202dd2fe166beab8a86d29465c41a0288cbe229fac60c1ab8d/jsonlines-4.0.0-py3-none-any.whl",
        "185b334ff2ca5a91362993f42e83588a360cf95ce4b71a73548502bda52a7c55",
        8701,
    ),
    (
        "https://files.pythonhosted.org/packages/76/79/524d5f481ffd5095dd8fdf273bc410214d1a2bb19e6c9a3f379071f1cfc4/jsonpatch-1.48-py3-none-any.whl",
        "389f25f006940b7bc939c33ca0f6cf8875527dc70599520652a269040ce40b63",
        27220,
    ),
    (
        "https://files.pythonhosted.org/packages/72/07/3c953fceee478798f6bdf7b39e0da0793f437e8b5f9e9e1360fba32fb25c/jsonpointer-3.12-py3-none-any.whl",
        "89f391335ee2719ae72149e1bf30a4d4c7c5d92f15a3f53faa8d20cc7c6c7f19",
        10562,
    ),
    (
        "https://files.pythonhosted.org/packages/d5/20/f2f5877cb22ee3c26c52f4be8737c7cb94e23e6a5e47bf05654b189ed0af/mmh3-5.3.1-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
        "3b037280edc7a609a987fa7661a1132a3f6d721f46b299ed5f9f641b35ab415a",
        103108,
    ),
    (
        "https://files.pythonhosted.org/packages/a0/c4/c2971a3ba4c6103a3d10c4b0f24f461ddc027f0f09763220cf35ca1401b3/nest_asyncio-1.6.0-py3-none-any.whl",
        "87af6efd6b5e897c81050477ef65c62e2b2f35d51703cae01aff2905b1852e1c",
        5195,
    ),
    (
        "https://files.pythonhosted.org/packages/bc/9c/15fb18bd708949dc953fe120be8d3c7a1b1b140c2592a9c81ce7961e9ecf/s3fs-2026.9.0-py3-none-any.whl",
        "eb185fd98e20a1c9375087d9b2a6ef506f0cd01ae0c0c94efc79f88403fb81fe",
        33126,
    ),
    (
        "https://files.pythonhosted.org/packages/e2/a4/88f9e10498759121060b8050a1554753dfc04df39bbbcc7aad9ecc8d9440/s3transfer-0.19.3-py3-none-any.whl",
        "24eb7535804306355e31c952a90022bf5b2f9d760f9a11062c7493a217c08553",
        90252,
    ),
    (
        "https://files.pythonhosted.org/packages/ab/60/27e29790ff15ea0a485b8dc200d514328294f956e4abc3d43d88c0b5338d/semver-3.1.2-py3-none-any.whl",
        "45d107c3fa880910441dfab9d555046cba2829a25feeadfe4a1566439635c510",
        19100,
    ),
    (
        "https://files.pythonhosted.org/packages/c0/44/21d6bf170bf40b41396480d8d49ad640bca3f2b02139cd52aa1e272830a5/shortuuid-1.0.13-py3-none-any.whl",
        "a482a497300b49b4953e15108a7913244e1bb0d41f9d332f5e9925dba33a3c5a",
        10529,
    ),
    (
        "https://files.pythonhosted.org/packages/d6/26/1ff2b0721ac66a3ec5b1402b333110b352ab0a8724052ac279a7b82d40c4/tenacity-9.2.1-py3-none-any.whl",
        "9e56f17539296baab7beabb08b92f6ee3d7be92d8be72d763360677c2ad6580e",
        32310,
    ),
    (
        "https://files.pythonhosted.org/packages/fb/be/35261223d9416a0751cdff1c7b4a6f881387218a12d439fe22fefebc8c04/textual-8.2.8-py3-none-any.whl",
        "267375fd402dc8d981457212efa71f0e3365fd17bba144ba9bb3ed7563cb374a",
        731418,
    ),
]

out_dir = Path(sys.argv[1])
for url, sha256, size in WHEELS:
    dest = out_dir / url.rsplit("/", 1)[-1]
    if dest.is_file() and dest.stat().st_size == size:
        if hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
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
    python3 -m pip install --no-deps --no-index --no-compile --target "$W/third_party/python" \
        "$W/third_party/wheels/aiobotocore-3.10.0-py3-none-any.whl" \
        "$W/third_party/wheels/aioitertools-0.13.0-py3-none-any.whl" \
        "$W/third_party/wheels/boto3-1.44.7-py3-none-any.whl" \
        "$W/third_party/wheels/botocore-1.44.7-py3-none-any.whl" \
        "$W/third_party/wheels/frozendict-2.5.1-py3-none-any.whl" \
        "$W/third_party/wheels/inspect_ai-0.3.150-py3-none-any.whl" \
        "$W/third_party/wheels/jsonlines-4.0.0-py3-none-any.whl" \
        "$W/third_party/wheels/jsonpatch-1.48-py3-none-any.whl" \
        "$W/third_party/wheels/jsonpointer-3.12-py3-none-any.whl" \
        "$W/third_party/wheels/mmh3-5.3.1-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl" \
        "$W/third_party/wheels/nest_asyncio-1.6.0-py3-none-any.whl" \
        "$W/third_party/wheels/s3fs-2026.9.0-py3-none-any.whl" \
        "$W/third_party/wheels/s3transfer-0.19.3-py3-none-any.whl" \
        "$W/third_party/wheels/semver-3.1.2-py3-none-any.whl" \
        "$W/third_party/wheels/shortuuid-1.0.13-py3-none-any.whl" \
        "$W/third_party/wheels/tenacity-9.2.1-py3-none-any.whl" \
        "$W/third_party/wheels/textual-8.2.8-py3-none-any.whl" \
        >/dev/null
    PYTHONPATH="$W/third_party/python" python3 -c \
        'from inspect_ai.scorer._common import match_str; print("[prepare] inspect_ai import ok")'
    touch "$W/third_party/python/.complete"
fi
