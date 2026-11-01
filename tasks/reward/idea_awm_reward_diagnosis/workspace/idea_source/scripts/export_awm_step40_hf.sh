#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CHECKPOINT:-/local/user/models/opsd-tool/awm_opsd_qwen3_8b_step693_wandb_v3/global_step_40}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/awm_opsd_qwen3_8b_step693_wandb_v3}"
PYTHON="${PYTHON:-/local/user/miniconda3/envs/verl-tool-env/bin/python}"
VERIFY_PYTHON="${VERIFY_PYTHON:-/local/user/conda_envs/seed/bin/python}"
ACTOR="$CHECKPOINT/actor"
TARGET="$OUTPUT_ROOT/$(basename "$CHECKPOINT")_hf"

[[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
[[ -x "$VERIFY_PYTHON" ]] || { echo "Python not executable: $VERIFY_PYTHON" >&2; exit 1; }
[[ -f "$ACTOR/config.json" && -f "$ACTOR/tokenizer.json" ]] || {
    echo "Missing model config or tokenizer in $ACTOR" >&2; exit 1;
}
for rank in {0..7}; do
    [[ -f "$ACTOR/model_world_size_8_rank_${rank}.pt" ]] || {
        echo "Missing model shard for rank $rank in $ACTOR" >&2; exit 1;
    }
done
if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
    [[ -d "$TARGET" ]] || { echo "Output missing: $TARGET" >&2; exit 1; }
else
    [[ ! -e "$TARGET" ]] || { echo "Output already exists: $TARGET" >&2; exit 1; }
    mkdir -p "$OUTPUT_ROOT"
    cd "$PROJECT_ROOT"
    OMP_NUM_THREADS=1 "$PYTHON" scripts/model_merger.py merge \
        --backend fsdp --local_dir "$ACTOR" --target_dir "$TARGET"
fi

for file in tokenizer.json tokenizer_config.json chat_template.jinja; do
    [[ -e "$TARGET/$file" ]] || cp "$ACTOR/$file" "$TARGET/$file"
done

"$VERIFY_PYTHON" - "$TARGET" "$ACTOR" <<'PY'
import sys
from pathlib import Path

import json

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

target = Path(sys.argv[1])
actor = Path(sys.argv[2])
source_config = json.loads((actor / "config.json").read_text())
export_config = json.loads((target / "config.json").read_text())
source_rope = source_config["rope_parameters"]["rope_theta"]
if "rope_theta" in export_config and export_config["rope_theta"] != source_rope:
    export_config["rope_theta"] = source_rope
    (target / "config.json").write_text(json.dumps(export_config, indent=2) + "\n")
config = AutoConfig.from_pretrained(target)
tokenizer = AutoTokenizer.from_pretrained(target)
weights = sorted(target.glob("*.safetensors"))
if not weights or not (target / "tokenizer.json").is_file():
    raise RuntimeError("Missing model weights or tokenizer")
index_path = target / "model.safetensors.index.json"
expected = None
if index_path.exists():
    expected = json.loads(index_path.read_text())["weight_map"]
    if {path.name for path in weights} != set(expected.values()):
        raise RuntimeError("Weight index does not match safetensors files")
count = 0
for path in weights:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = handle.keys()
        if not keys:
            raise RuntimeError(f"Empty weights file: {path}")
        for key in keys:
            if handle.get_slice(key).get_shape() == []:
                raise RuntimeError(f"Unexpected scalar weight: {key}")
            if expected is not None and expected.get(key) != path.name:
                raise RuntimeError(f"Missing or misplaced tensor in weight index: {key}")
            count += 1
if expected is not None and count != len(expected):
    raise RuntimeError("Weight index tensor count does not match model files")
if config.model_type != "qwen3" or count < 100 or len(tokenizer) == 0:
    raise RuntimeError("Model config, weights, or tokenizer failed validation")
print(f"Verified {count} tensors in {len(weights)} safetensors file(s): {target}")
PY
