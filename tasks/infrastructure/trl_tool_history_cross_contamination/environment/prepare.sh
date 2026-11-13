#!/usr/bin/env bash
# trl_tool_history_cross_contamination — prepare stage.
#
# Runs in the prepare-stage container: root, the verl-vllm0.24 image, the
# only networked stage, and $AUTORLBENCH_WORKSPACE writable. It shadows the
# recorded pure-Python pins, fetches the Qwen3-4B weights, rebuilds the py3.9
# case-venv (the pinned pytest 3.10.1 stack the corpus runner executes
# thefuck's tests with — the tool-rl image baked it at /opt/case-venv; on
# this image it lives inside the workspace so it survives the stage
# handoff), pip-installs the vendored TRL checkout, and fetches the pinned
# upstream corpus snapshots.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"

# --- image-level env, seeded here --------------------------------------------
# The verl-vllm0.24 image bakes no CUDA_DEVICE_ORDER / TOKENIZERS_PARALLELISM
# defaults; the two entries the recording rig's image carried ride the
# prepared/env handoff into the agent/rerun/verifier stages via the activate
# hook. Idempotent.
for _kv in CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false; do
    if ! grep -qs "^${_kv%%=*}=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
        echo "$_kv" >> "${AUTORLBENCH_PREPARED}/env"
    fi
done

# --- recorded pure-Python pins, shadowed over the image ----------------------
# The task runs on the packaged verl-vllm0.24 image (torch 2.11.0+cu130 /
# vLLM 0.24.0 / flashinfer 0.6.12 / py3.12): the compiled layer matches the
# recorded tool-rl stack, including the actively-used engine (GRPOTrainer
# runs use_vllm=True, vllm_mode=server — vLLM 0.24.0 is the recorded
# version, so the vendored TRL 1.15.0.dev0 talks to the same server API).
# The image's pure-Python layer is frozen at different versions, so the
# recorded majors are shadowed from $AUTORLBENCH_PREPARED/pysite;
# tokenizers (0.22.2) already matches the image. --no-deps: the image
# carries the transitive closure, and letting pip resolve would float pins
# the image deliberately holds. Native drift the shadow cannot reach (ray
# 2.56.1 vs 2.60.0, nccl 2.30.7 vs 2.28.9, pyarrow 25.0.0 vs 26.0.1, numpy
# 1.26.4 vs 2.3.5) is accepted — the defective run was pilot-verified on
# this image (2026-11-13); the full reference-evidence re-record on this
# image is pending.
site="$AUTORLBENCH_PREPARED/pysite"
if [ ! -d "$site/transformers" ]; then
    _pip_args=(--no-cache-dir --no-deps --only-binary :all: --target "$site")
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" \
        transformers==5.10.1 accelerate==1.15.0 datasets==5.1.0 peft==0.21.2 \
        wandb==0.31.0 huggingface_hub==1.33.0
fi
# Shadow for the rest of this script and for the agent/rerun/verifier stages
# via prepared/env (run.sh re-prefixes it ahead of $PWD/vendor/trl). The
# vLLM server subprocess inherits PYTHONPATH too, so trainer and server see
# the same shadowed transformers, as in the recorded environment.
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
if ! grep -qs "^PYTHONPATH=" "${AUTORLBENCH_PREPARED}/env" 2>/dev/null; then
    echo "PYTHONPATH=$site:$W/vendor/trl:$W" >> "${AUTORLBENCH_PREPARED}/env"
fi
PYTHONPATH="$site" python3 - <<'EOF'
import transformers, accelerate, datasets, peft, wandb, huggingface_hub
assert transformers.__version__ == "5.10.1", transformers.__version__
assert accelerate.__version__ == "1.15.0", accelerate.__version__
assert datasets.__version__ == "5.1.0", datasets.__version__
assert peft.__version__ == "0.21.2", peft.__version__
assert wandb.__version__ == "0.31.0", wandb.__version__
assert huggingface_hub.__version__ == "1.33.0", huggingface_hub.__version__
EOF

# --- py3.9 case-venv (the corpus runner's test interpreter) ------------------
# The tool-rl image baked /opt/case-venv: uv-managed CPython 3.9.25 plus the
# pinned pre-modern test stack thefuck's snapshot tests run under (pytest
# 3.10.1 era). This image has no such layer and container filesystems do
# not survive the stage handoff, so the venv is rebuilt inside the
# workspace — run.sh points --test-python at it. uv's python-build-standalone
# download needs GitHub access, same as the corpus fetches below.
if [ ! -x "$W/case-venv/bin/python" ]; then
    _pip_args=(--no-cache-dir)
    if [ -n "${AUTORLBENCH_PIP_INDEX_URL:-}" ]; then
        _pip_args+=(--index-url "$AUTORLBENCH_PIP_INDEX_URL")
        if [ -n "${AUTORLBENCH_PIP_TRUSTED_HOST:-}" ]; then
            _pip_args+=(--trusted-host "$AUTORLBENCH_PIP_TRUSTED_HOST")
        fi
    fi
    python3 -m pip install "${_pip_args[@]}" --break-system-packages uv==0.9.9
    export UV_PYTHON_INSTALL_DIR="$W/legacy-python"
    uv python install 3.9.25
    uv venv --python 3.9.25 "$W/case-venv"
    uv pip install --python "$W/case-venv/bin/python" \
        atomicwrites==1.4.1 \
        attrs==23.2.0 \
        certifi==2026.10.23 \
        charset-normalizer==3.5.2 \
        colorama==0.4.6 \
        decorator==4.4.2 \
        exceptiongroup==1.3.1 \
        idna==3.20 \
        iniconfig==2.1.0 \
        mock==5.2.0 \
        more-itertools==10.8.0 \
        nose==1.3.7 \
        packaging==26.3 \
        pexpect==4.9.0 \
        pluggy==1.6.0 \
        psutil==7.2.2 \
        ptyprocess==0.7.0 \
        py==1.11.0 \
        pyte==0.8.2 \
        pytest==3.10.1 \
        pytest-mock==1.10.4 \
        requests==2.32.5 \
        six==1.17.0 \
        setuptools==58.0.4 \
        tomli==2.5.0 \
        tqdm==4.67.1 \
        typing_extensions==4.16.0 \
        urllib3==2.6.3 \
        wcwidth==0.9.4
fi
"$W/case-venv/bin/python" --version 2>&1 | grep -q "3\.9\."

autorlbench_fetch_hf \
    --repo Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --allow .gitattributes \
    --expect .gitattributes=34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930 \
    --expect-size .gitattributes=1570 \
    --allow LICENSE \
    --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
    --expect-size LICENSE=11343 \
    --allow README.md \
    --expect README.md=71add1cd091c309b1cb1a6b943f2476eb2b004124b3bd1d32dcd0b24cc1abc47 \
    --expect-size README.md=16857 \
    --allow config.json \
    --expect config.json=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a \
    --expect-size config.json=726 \
    --allow generation_config.json \
    --expect generation_config.json=2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 \
    --expect-size generation_config.json=239 \
    --allow merges.txt \
    --expect merges.txt=8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 \
    --expect-size merges.txt=1671853 \
    --allow model-00001-of-00003.safetensors \
    --expect model-00001-of-00003.safetensors=328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223 \
    --expect-size model-00001-of-00003.safetensors=3957900840 \
    --allow model-00002-of-00003.safetensors \
    --expect model-00002-of-00003.safetensors=6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5 \
    --expect-size model-00002-of-00003.safetensors=3987450520 \
    --allow model-00003-of-00003.safetensors \
    --expect model-00003-of-00003.safetensors=e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1 \
    --expect-size model-00003-of-00003.safetensors=99630640 \
    --allow model.safetensors.index.json \
    --expect model.safetensors.index.json=6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2 \
    --expect-size model.safetensors.index.json=32819 \
    --allow tokenizer.json \
    --expect tokenizer.json=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 \
    --expect-size tokenizer.json=11422654 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 \
    --expect-size tokenizer_config.json=9732 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$AUTORLBENCH_WORKSPACE/assets/model"
python3 -m pip install --no-deps --no-build-isolation --break-system-packages -e "$AUTORLBENCH_WORKSPACE/vendor/trl"
python3 -m compileall -q "$AUTORLBENCH_WORKSPACE/reproductions"

# Pinned upstream corpus snapshots.
mkdir -p "$AUTORLBENCH_WORKSPACE/.corpus_archives"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/2ced7a7f33ae0bec3ffc7a43ce95330bdf6cfcb9" --sha256 30e75ae72f14707a80ac03a5a2d36f371879adeb2638dca9116fb155fdd6369c --size 1382930 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-2ced7a7f33ae0bec3ffc7a43ce95330bdf6cfcb9.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/7c858fadb3458be829d3d43666ccb46c3ed5b8a0" --sha256 110b9403ab0f59e9db458a75122b94adae0e8288baf00114004c121a1c3a30a0 --size 1364662 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-7c858fadb3458be829d3d43666ccb46c3ed5b8a0.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/797ca1c5647c565f62e21a8e29515c8b0fbe275f" --sha256 7cd4799f9a166b5a08ba5e70ea81dc7cd9e9da0ba5a33147875c5c5c1a12203b --size 1364624 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-797ca1c5647c565f62e21a8e29515c8b0fbe275f.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/64d6835e15a22cf8803d398cbb593f748c550e8a" --sha256 9517fe46d71953715fbb7a030c624cd9202830a01ee41b48ee26cf90e3a5dc2a --size 1360418 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-64d6835e15a22cf8803d398cbb593f748c550e8a.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/449cb9a00693c8b4d97d5fda8d732cf0978e117e" --sha256 6fc2f2eb8fba1f022f2742ef968ac11428f63038b9aa55adcfc34dc60f06e30a --size 1355292 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-449cb9a00693c8b4d97d5fda8d732cf0978e117e.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/ce6b82c92d78ae283cb3db001766b76f6647bc47" --sha256 8dd8aa1c553c8602b01d098292cd971721961df0edc48ab9bcd1a2fcfb90cdcd --size 798748 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-ce6b82c92d78ae283cb3db001766b76f6647bc47.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/8bd6c5da67e55c64257345efa4e3cc454c42475c" --sha256 19cab7dd4222d79248a88f3f9ea46ff0aa288ed7034e925ddd34368dd0fe22d8 --size 798561 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-8bd6c5da67e55c64257345efa4e3cc454c42475c.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/92f3c8fb52b32b79005b4864c31a5c2d8c45f4b1" --sha256 344bf890ef7b5a3f62f8c4d583f5505ee8cd123b3be2e6835b712644a006447c --size 798272 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-92f3c8fb52b32b79005b4864c31a5c2d8c45f4b1.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/2af65071d84a7d1d14a4126364d9b4c9b5241f3c" --sha256 193e40e8bbdbf2c6d5ee6033e7bc46e4a949ca94086e64aef2cd92182720d032 --size 786456 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-2af65071d84a7d1d14a4126364d9b4c9b5241f3c.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/3a39deb485995e67afb1919972cd1c9aaedf4c32" --sha256 c99310bfb6bd07a5e6836e32cff0076ec028fd466a44363c9c1e65541308aa0a --size 783508 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-3a39deb485995e67afb1919972cd1c9aaedf4c32.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/b65a9a0a4fd9bef394b45a1d367d29aa1e1c403e" --sha256 518c550154d614f489cf79350de4d7553809445c4dedf90fd85547d61bf08377 --size 777268 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-b65a9a0a4fd9bef394b45a1d367d29aa1e1c403e.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/959b96cf6ec8cedda05dc58efe0e0f3bd6ed2f4e" --sha256 12a864083eb1af878ed0834f2017ea0159ea16fd2da96a1d05a39bc10d3ecb78 --size 776005 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-959b96cf6ec8cedda05dc58efe0e0f3bd6ed2f4e.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/71dc2666ccf62e653291d9a7a08e2c6c3320425b" --sha256 f3f038ae41b745bd45f6e9dc7f678539774aee57bb4b454fc9865d90295476e6 --size 773247 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-71dc2666ccf62e653291d9a7a08e2c6c3320425b.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/42a8b4f639269886e468762e6d100b6f01aad8ab" --sha256 26c63c2d7e22625a6d56fb9c9f256d20035a81274f3f96ac6c55c38fd998d08f --size 766461 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-42a8b4f639269886e468762e6d100b6f01aad8ab.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/tqdm/tqdm/tar.gz/08b8ad1ff3bd003ef8309faaa0cc108ffa40317d" --sha256 08074f3a3b45dadc51fdbee814bc8a8f7c33ab62b4f7252caa14ad514228362a --size 467042 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/tqdm_tqdm-08b8ad1ff3bd003ef8309faaa0cc108ffa40317d.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/tqdm/tqdm/tar.gz/9da1d5d116aec7a23d8f6dc22d5e23ecb1c40a7c" --sha256 1c88a417588cd1b58e4bbbc65299b8fae556c07b1532871c67a9ca286ee7811a --size 74819 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/tqdm_tqdm-9da1d5d116aec7a23d8f6dc22d5e23ecb1c40a7c.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/0a6a3db65d2fc480c5b2f1135137f34c9f06b742" --sha256 c9246e36e2b9754a89d9cbd51b226002409198f1bac46a6893a37eb48dd37d5b --size 775721 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-0a6a3db65d2fc480c5b2f1135137f34c9f06b742.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/7cb0388ed0845545e878b29783bbf8e901a02745" --sha256 edde185f2660ff558b5735142c25b9de66c4f9bba98dd1b90952bd78ff79cb7f --size 766891 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-7cb0388ed0845545e878b29783bbf8e901a02745.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/bc6b107066d3f1e60b4cfcaa8cf6399e98cf1b1c" --sha256 553eb90a509373e918abd08ff0effb86e7db2befdb49b0ca20b8d296bc8085de --size 765892 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-bc6b107066d3f1e60b4cfcaa8cf6399e98cf1b1c.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/cb33c912e5f2f4c2da6b70d708ff0437bfcd3b94" --sha256 dea0f79d988ffbfd1b4a395c1e27f64835a1c714533dedba1429a53b18cd180e --size 32328 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-cb33c912e5f2f4c2da6b70d708ff0437bfcd3b94.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/40ab4eb62db57627bff10cf029d29c94704086a2" --sha256 f93610e8d9c614c2a5842beebb714e04556672b686e5e89224d06783a52d9340 --size 1371317 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-40ab4eb62db57627bff10cf029d29c94704086a2.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/ac343fb1bd7dadbb9e1a9fd7e3071f5778e338a4" --sha256 6e7e84ad94758ee0420529dccf1a04a6acbe3b9358e1a02a7a567d753fd97d00 --size 1369346 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-ac343fb1bd7dadbb9e1a9fd7e3071f5778e338a4.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/68949a592248913b52bfd50036893553153fcddb" --sha256 2175d519ffc0bc49cbd6d20f5608d15c356771295badfa2017c07621cd50fd60 --size 1365858 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-68949a592248913b52bfd50036893553153fcddb.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/183b70c8b8885843efefd2bd4e74dc0a7d42d173" --sha256 6b0e7a4bdbd3658b5ac092a8f724e5f15b22053064c7b868bfdeaf4ddd2478f9 --size 784119 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-183b70c8b8885843efefd2bd4e74dc0a7d42d173.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/faa7ee603057fa98c25507d30180c055d10d13d4" --sha256 d0a476737e12a5af4c6ef7b9c14e8eb65dfb561d817bedcd73838c2eae3fcf3b --size 771745 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-faa7ee603057fa98c25507d30180c055d10d13d4.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/4129ff2717cc6e6fa51d70cc4e6c31d56ef8e2c9" --sha256 d7a46276c8c309b68a2e1f62142dea56a3cddbcb89e711ecc441efa2c7564b5e --size 771583 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-4129ff2717cc6e6fa51d70cc4e6c31d56ef8e2c9.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/12394ca8423a438915fed996383b44471fc1139d" --sha256 2062841a6113e6ad551892223d7ddc4a86185ae6d3ba85dda16f6ddfa01fa8bc --size 770656 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-12394ca8423a438915fed996383b44471fc1139d.tar.gz"
autorlbench_fetch_url --url "https://codeload.github.com/nvbn/thefuck/tar.gz/4a2f869c6d6ef03d8d7ada1121cc6631d7ef979e" --sha256 fb1b453a68ec1962e0846dbafb0c110f108f9f1aa49c5003ff38792613b110df --size 765414 --dest "$AUTORLBENCH_WORKSPACE/.corpus_archives/nvbn_thefuck-4a2f869c6d6ef03d8d7ada1121cc6631d7ef979e.tar.gz"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/444908ce1c17767ef4aaf9e0b4950497914f7f63/tests/rules/test_pip_unknown_command.py" --sha256 4365ddcc8e2600fcb2e7337eb954c5abab3e51647fb267f0afe351eca6b8f8af --size 1085 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-1/tests/rules/test_pip_unknown_command.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/c205683a8df8a57e2db1e9816a5a7ce3255b08fc/tests/rules/test_git_push.py" --sha256 ff4aaaceeeb8d770ccc3fe06bf85ec65bbf8bd2352aa9dc2b95fff6b9c8492f0 --size 2396 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-5/tests/rules/test_git_push.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/7c858fadb3458be829d3d43666ccb46c3ed5b8a0/tests/rules/test_git_branch_exists.py" --sha256 0acc278fddf8ee5362b7c9a04cfa21808c4321086f0658db43aebb705e8fa204 --size 1463 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-6/tests/rules/test_git_branch_exists.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/75d2c43997ca703150cbdb4c46ed7b2e2e71fd11/tests/rules/test_php_s.py" --sha256 9936616e5b1213084a05ccaa3d32a7c2a2198d385e2767f11e38366e21d0d4d1 --size 761 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-7/tests/rules/test_php_s.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/be48f027847161f907def8987706041c65a1fd58/tests/rules/test_dnf_no_such_command.py" --sha256 cf7c78e12805248a3e9f72451da0b399c17e6b423f8784f129b0d0d514718161 --size 8968 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-8/tests/rules/test_dnf_no_such_command.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/feb36ede5c518fdc3b6eddf945b2d8b1e2294d15/tests/rules/test_git_push.py" --sha256 cba367bcf88dc0433d190bad5e9149acced2bf5c67fbad7ed442bbd2988f7e6b --size 1219 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-9/tests/rules/test_git_push.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/0c84eefa55fc1b4bc4940b41d74568884344e35c/tests/rules/test_man.py" --sha256 8dbf7b466b58a20afe9dc983fa2debe7f2d4a290c325cdd04b0daa35393ece5e --size 1122 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-10/tests/rules/test_man.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/db7dffdb44ae5c7be8de088765463fbda96197d1/tests/rules/test_git_push.py" --sha256 a524f3bfd62d76889522457d1fe6197a312db13d553403d7ea92f5aae09d3b1c --size 1101 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-11/tests/rules/test_git_push.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/237bc579994de633fe104714156ddfa925a50b6e/tests/rules/test_git_branch_exists.py" --sha256 1b53b9affaddf0c9b961ea04f6a4b8bb41fd0979e28000d279003d4ddac28aa1 --size 1187 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-13/tests/rules/test_git_branch_exists.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/41707b80c61acadb7c87b0efcbf10f4186dc5937/tests/rules/test_git_add.py" --sha256 8e16f9ec28a62f8e260aff3889f69d7f863ad2c8294883364c71c04e6fcc0bc2 --size 1058 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-15/tests/rules/test_git_add.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/c3b1ba763708b8faaaf55717c436c4cd4c57a7ea/tests/rules/test_sudo.py" --sha256 d2666ef059d14c359f136b92646abc8c2ff7f268016e6ecb56648e2dd566b9f6 --size 1038 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-18/tests/rules/test_sudo.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/dc23d67a42dad54308a753639edd1ea0d15cb2e7/tests/rules/test_git_push_force.py" --sha256 30d3b6d00934930314ce3dc7855d9eaeb21fd2b8af644da3dbff46c6555415b6 --size 1923 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-19/tests/rules/test_git_push_force.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/213791d3c2af379ffa37a140735998736b41912e/tests/rules/test_git_fix_stash.py" --sha256 aab0c6f8ff69d9cfed8c6ee673011d6fd8c5fdc62aa99ed302b75fecf5695815 --size 1127 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-21/tests/rules/test_git_fix_stash.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/298c04f89c081dc16c8653aa017ca85dd14bfad6/tests/rules/test_mkdir_p.py" --sha256 e82cb7352cd83e0027d981f31222c502090b1048ba8f726249ea5311ecf7d415 --size 1201 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-25/tests/rules/test_mkdir_p.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/tqdm/tqdm/cae9d139c6df5614be3bf6e25ccbd600ee3286dc/tqdm/tests/tests_tqdm.py" --sha256 f92ff31cfc8f5f1428c460b887e33a44030f752070bdee43670424c03ead3f62 --size 41058 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/tqdm-8/tqdm/tests/tests_tqdm.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/tqdm/tqdm/d877c1dfb4739852105f7b967a8fceb869ac5042/tqdm/tests/tests_tqdm.py" --sha256 ab9da9a5e0b30fe8bac87c373ccfe91398d20b790afe3543eaf635b791d91c14 --size 7302 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/tqdm-9/tqdm/tests/tests_tqdm.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/280751b36e715b006c631ba6c08de99ccc74f6d2/tests/rules/test_dirty_unzip.py" --sha256 2973e5339b1cf3732766dd9cbc831f565b209abb06aa04c47504c1ec811975b7 --size 1402 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-20/tests/rules/test_dirty_unzip.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/feb3eee2a08f0cba4552373d728509bc90b561ab/tests/rules/test_vagrant_up.py" --sha256 3e53f49937b2e762739c9c92e15c987a45cd0fe0971f9ee99ce74bd8d79ba1f0 --size 2331 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-26/tests/rules/test_vagrant_up.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/1becd92b126a368d6e7d93aa8eea209414ce4aa2/tests/rules/test_open.py" --sha256 90d0add28e077db40d093cd79a4cddf0471cc48c0dd6e5b39c3674db59cd0ae3 --size 1194 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-27/tests/rules/test_open.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/25cc98a21a3450a046caf418f08713c82a290805/tests/rules/test_ls_lah.py" --sha256 5a38c8830404423047e290464243b2d335a1a46a0601fe55dc8e0e8458f2d34f --size 586 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-32/tests/rules/test_ls_lah.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/78ef9eec88f43d5727986be2237f6e0e250cbbbc/tests/test_utils.py" --sha256 5398f82f3da7f53a1143310e66710bba7539bc8667d7bf33c8b14aa8c370535b --size 8269 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-2/tests/test_utils.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/ce5feaebf7fd5c1190b5e14fbc1e962cc8db5f39/tests/shells/test_fish.py" --sha256 03645fc9b01f96d3ab434cc8b91235b926b9047441b834402db0e5f8b02ef667 --size 5241 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-3/tests/shells/test_fish.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/8db3cf604865e559090412ce80b0640e290ad83a/tests/shells/test_fish.py" --sha256 2604b7611f76427d2a97aed28b9e25a4c04f30754c881d0f89605e604f144784 --size 5009 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-4/tests/shells/test_fish.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/db6053b301e2b3f4363401e457b5dc4ad2e8429b/tests/shells/test_fish.py" --sha256 edbb81a374f658ea5cd4dec222a040c77a2bebfbe317b520ba8f99764ec35b7e --size 3021 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-14/tests/shells/test_fish.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/e2e8b6fc865452b4cfc1bed70e5b9b49807258ae/tests/test_types.py" --sha256 b34f2b9729d6722ef4491d035401c1ed5e3ca60db0cbcd22fd677a597d818b1c --size 2416 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-22/tests/test_types.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/9a02e821cdc58a4aba2c0acc521fb25cacab87a5/tests/test_utils.py" --sha256 6bc9a3ad387e449a13b76ccbba64900714b845b8306fc3d36738b0526e426b58 --size 5538 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-23/tests/test_utils.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/5d74344994da89ed01afd448f1c9d86b85e85351/tests/test_types.py" --sha256 ba77cfc007fb6bd04d05288f2b52ea1abbbf94a23aca25ae4168e4f47b692e5e --size 2313 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-24/tests/test_types.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/88831c424f569e6a55fc98883d3eeecc7d425b18/tests/test_types.py" --sha256 8c4d92a4f83a17092d4113fa52c1da9362ae876aa308ade234ffd295bb718a2a --size 621 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-29/tests/test_types.py"
autorlbench_fetch_url --url "https://raw.githubusercontent.com/nvbn/thefuck/88831c424f569e6a55fc98883d3eeecc7d425b18/tests/test_utils.py" --sha256 fc0974b4a9c1d0a8266cb60ab1e9d9cd8dffaca2752c275509092dfdff0055ab --size 4539 --dest "$AUTORLBENCH_WORKSPACE/.corpus_tests/thefuck-29/tests/test_utils.py"
python "$AUTORLBENCH_WORKSPACE/build_corpus.py"
