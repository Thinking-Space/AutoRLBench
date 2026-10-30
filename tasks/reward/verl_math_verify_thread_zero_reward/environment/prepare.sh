#!/usr/bin/env bash
# verl_math_verify_thread_zero_reward: prepare stage.
#
# Runs in a separate prepare-stage container: root, the `verl` image, **the only networked stage**, and
# $AUTORLBENCH_WORKSPACE is writable. It fetches the pinned model, GSM8K, MATH-lighteval and MATH-500, installs
# math-verify (not part of the `verl` image) into third_party/python from pinned wheels, and builds the parquet
# files exactly as the recorded runs did: verl's examples/data_preprocess/{gsm8k,math_dataset}.py from the
# workspace checkout, plus tools/prepare_math500.py.
#
# Destinations are what run_grpo.sh reads: MODEL=$WS/models/Qwen2.5-1.5B-Instruct,
# DATA=$WS/data/{gsm8k/train,gsm8k/test,math/train,math500/test}.parquet, PYTHONPATH+=$WS/third_party/python.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# --- Base model ---------------------------------------------------------------
autorlbench_fetch_hf \
    --repo Qwen/Qwen2.5-1.5B-Instruct \
    --type model \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --allow LICENSE \
    --expect LICENSE=832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e \
    --expect-size LICENSE=11343 \
    --allow README.md \
    --expect README.md=2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada \
    --expect-size README.md=4917 \
    --allow config.json \
    --expect config.json=98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670 \
    --expect-size config.json=660 \
    --allow generation_config.json \
    --expect generation_config.json=e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6 \
    --expect-size generation_config.json=242 \
    --allow merges.txt \
    --expect merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 \
    --expect-size merges.txt=1671839 \
    --allow model.safetensors \
    --expect model.safetensors=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expect-size model.safetensors=3087467144 \
    --allow tokenizer.json \
    --expect tokenizer.json=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
    --expect-size tokenizer.json=7031645 \
    --allow tokenizer_config.json \
    --expect tokenizer_config.json=5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 \
    --expect-size tokenizer_config.json=7305 \
    --allow vocab.json \
    --expect vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 \
    --expect-size vocab.json=2776833 \
    --dest "$W/models/Qwen2.5-1.5B-Instruct"

# --- Datasets -----------------------------------------------------------------
autorlbench_fetch_hf \
    --repo openai/gsm8k \
    --type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --allow README.md \
    --expect README.md=a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1 \
    --expect-size README.md=7935 \
    --allow main/train-00000-of-00001.parquet \
    --expect main/train-00000-of-00001.parquet=ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d \
    --expect-size main/train-00000-of-00001.parquet=2306545 \
    --allow main/test-00000-of-00001.parquet \
    --expect main/test-00000-of-00001.parquet=ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 \
    --expect-size main/test-00000-of-00001.parquet=419088 \
    --dest "$W/data/raw/gsm8k"

# All eight configs are fetched: `datasets` resolves every config listed in the README, not only `default`.
autorlbench_fetch_hf \
    --repo DigitalLearningGmbH/MATH-lighteval \
    --type dataset \
    --revision 0530c78699ea5e8eb5530600900e1f328b48acad \
    --allow README.md \
    --expect README.md=3bb056303034d0f49925396ede385ee2feed638c26e2abbf3a8ff0b9976a9bc4 \
    --expect-size README.md=8412 \
    --allow algebra/test-00000-of-00001.parquet \
    --expect algebra/test-00000-of-00001.parquet=82afa924bb3f78d48b8c91296f0b988f4a27d984b55722ce058dcf38581973af \
    --expect-size algebra/test-00000-of-00001.parquet=353114 \
    --allow algebra/train-00000-of-00001.parquet \
    --expect algebra/train-00000-of-00001.parquet=3f4f43beb2db5cacad772d30ad9eda027141a2abeff9aad08cf02d254d46175a \
    --expect-size algebra/train-00000-of-00001.parquet=505186 \
    --allow counting_and_probability/test-00000-of-00001.parquet \
    --expect counting_and_probability/test-00000-of-00001.parquet=d2d9564b7af4dcffa5bef1df13f90de648a5ac497342e2ce12bca9eec8aeabde \
    --expect-size counting_and_probability/test-00000-of-00001.parquet=175032 \
    --allow counting_and_probability/train-00000-of-00001.parquet \
    --expect counting_and_probability/train-00000-of-00001.parquet=8bd338b2bb20e41bc3c70daa36ff93f30faee106ced6dc788d292e37eddce98b \
    --expect-size counting_and_probability/train-00000-of-00001.parquet=329354 \
    --allow data/test-00000-of-00001.parquet \
    --expect data/test-00000-of-00001.parquet=7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6 \
    --expect-size data/test-00000-of-00001.parquet=1855870 \
    --allow data/train-00000-of-00001.parquet \
    --expect data/train-00000-of-00001.parquet=eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933 \
    --expect-size data/train-00000-of-00001.parquet=2992151 \
    --allow geometry/test-00000-of-00001.parquet \
    --expect geometry/test-00000-of-00001.parquet=580ea70aa18c593ee7e2492623e6e42281d2db27b8ba7de8bc903ac2b6ca0865 \
    --expect-size geometry/test-00000-of-00001.parquet=263835 \
    --allow geometry/train-00000-of-00001.parquet \
    --expect geometry/train-00000-of-00001.parquet=b31a8b3eb50e509b4c6b704ccd59646ca70efb306c1215a64c4e5d308a835c61 \
    --expect-size geometry/train-00000-of-00001.parquet=549388 \
    --allow intermediate_algebra/test-00000-of-00001.parquet \
    --expect intermediate_algebra/test-00000-of-00001.parquet=0b69a6bc6216fe7997dd8399478b6feffc2fba4fc5a4e255228b0ed265dbbade \
    --expect-size intermediate_algebra/test-00000-of-00001.parquet=394797 \
    --allow intermediate_algebra/train-00000-of-00001.parquet \
    --expect intermediate_algebra/train-00000-of-00001.parquet=d18b4ed64f77e3e45d7d7c4eb386a6bc0cc3ade39e5926938caf4d817acf07a6 \
    --expect-size intermediate_algebra/train-00000-of-00001.parquet=575154 \
    --allow number_theory/test-00000-of-00001.parquet \
    --expect number_theory/test-00000-of-00001.parquet=41695431c8c9f54c367f276b369586050b25fc86ba01d86468c13e050c397d67 \
    --expect-size number_theory/test-00000-of-00001.parquet=182003 \
    --allow number_theory/train-00000-of-00001.parquet \
    --expect number_theory/train-00000-of-00001.parquet=43145931797589e5ae252eabe747a8e447314d1498aa0f2e1c9ce781a514cea2 \
    --expect-size number_theory/train-00000-of-00001.parquet=308653 \
    --allow prealgebra/test-00000-of-00001.parquet \
    --expect prealgebra/test-00000-of-00001.parquet=2aa4f6c0d0f53432e58f91aff18a27f6dcb83e668d963f2d5c0bc2e5df9023b6 \
    --expect-size prealgebra/test-00000-of-00001.parquet=267729 \
    --allow prealgebra/train-00000-of-00001.parquet \
    --expect prealgebra/train-00000-of-00001.parquet=ef1622b40248ac858711ba61f6c76e0a4f1ae9b090a6a17e5cab7dcac53c9b05 \
    --expect-size prealgebra/train-00000-of-00001.parquet=383626 \
    --allow precalculus/test-00000-of-00001.parquet \
    --expect precalculus/test-00000-of-00001.parquet=1f5772bfcead8206f0645536e02dbe6e275371f9e2bbb3d45371f6946acb16bb \
    --expect-size precalculus/test-00000-of-00001.parquet=242241 \
    --allow precalculus/train-00000-of-00001.parquet \
    --expect precalculus/train-00000-of-00001.parquet=c51572f7b5a4e1a18a7898c3f7bc111015ff5fa14dc9274d0c639a6beb23f1ed \
    --expect-size precalculus/train-00000-of-00001.parquet=353745 \
    --dest "$W/data/raw/MATH-lighteval"

autorlbench_fetch_hf \
    --repo HuggingFaceH4/MATH-500 \
    --type dataset \
    --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
    --allow README.md \
    --expect README.md=5d32f78943f75423da61419c5bb98cee964d5a29183b11a5294f661fb96f69f7 \
    --expect-size README.md=412 \
    --allow test.jsonl \
    --expect test.jsonl=35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132 \
    --expect-size test.jsonl=446564 \
    --dest "$W/data/raw/MATH-500"

# --- math-verify (used by reward/mixed_math_reward.py) --------------------------
autorlbench_fetch_url \
    --url https://files.pythonhosted.org/packages/62/76/6b4969bccc842b6567f7e6ee015684b9428a9b7fcbdf479e73716f43597f/math_verify-0.9.0-py3-none-any.whl \
    --sha256 3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194 \
    --size 29967 \
    --dest "$W/third_party/wheels/math_verify-0.9.0-py3-none-any.whl"
autorlbench_fetch_url \
    --url https://files.pythonhosted.org/packages/e9/61/f75cd1fa54d8434276126034aed54dd120747de9a8fa013cdd79545ccbeb/latex2sympy2_extended-1.11.0-py3-none-any.whl \
    --sha256 aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234 \
    --size 209050 \
    --dest "$W/third_party/wheels/latex2sympy2_extended-1.11.0-py3-none-any.whl"
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
