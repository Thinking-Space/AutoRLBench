#!/bin/bash
# ============================================================
# Convert mix jsonl -> sstable using lizhen's zyx_infra tool.
#
# trans_sstable.py only accepts {input,output} or {messages} schema,
# our raw uses {question,response}, so we rename keys on-the-fly first.
#
# Run on a node that has internet access (or with cached wheels):
#   bash scripts/data/0415template/convert_to_sstable.sh
#
# Output: /user/luozhenghang/0226_miles/data/sft/0415template/sft_data_mix4k_union.qwen3_8b.0415template/sft_data_mix4k_union.qwen3_8b.0415template.sstable
# ============================================================
set -euo pipefail

ZYX=/user/lizhen/repos/zyx_infra
SRC_JSONL=/user/luozhenghang/0226_miles/data/sft/0415template/raw/sft_data_mix4k_union.sample_6264.jsonl
ADAPTED=/user/luozhenghang/0226_miles/data/sft/0415template/raw/sft_data_mix4k_union.sample_6264.adapted.jsonl
OUT_DIR=/user/luozhenghang/0226_miles/data/sft/0415template/sft_data_mix4k_union.qwen3_8b.0415template
OUT_SST=${OUT_DIR}/sft_data_mix4k_union.qwen3_8b.0415template.sstable

# 0) regenerate jsonl if missing
if [ ! -f "${SRC_JSONL}" ]; then
  echo "[INFO] mix jsonl missing, rebuild from lizhen raw..."
  python3 "$(dirname "$0")/prepare_mix.py"
fi

mkdir -p "${OUT_DIR}"

# 1) rename keys: {question, response} -> {input, output}
python3 - <<PY
import json
src = "${SRC_JSONL}"; dst = "${ADAPTED}"
n = 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fo.write(json.dumps({"input": r["question"], "output": r["response"]}, ensure_ascii=False) + "\n")
        n += 1
print(f"adapted {n} rows -> {dst}")
PY

# 2) install deps if missing (one-shot)
python3 -c "import thriftpy2, modelbest_sdk" 2>/dev/null || \
  pip3 install --quiet thriftpy2 modelbest_sdk numpy torch transformers==4.41 sentencepiece protobuf termcolor

# 3) run converter
cd "${ZYX}"
python3 trans_sstable.py \
  --json-file "${ADAPTED}" \
  --output-sstable "${OUT_SST}" \
  --tokenizer qwen3

echo "==============================================="
echo "DONE -> ${OUT_SST}"
ls -la "${OUT_SST}"
echo "==============================================="
