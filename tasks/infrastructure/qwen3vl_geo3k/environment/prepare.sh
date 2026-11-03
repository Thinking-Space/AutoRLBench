#!/usr/bin/env bash
# qwen3vl_geo3k — prepare stage.
#
# Runs in a separate prepare-stage container: root, the task's declared image,
# **the only stage with network**. In the agent stage this file is mounted
# read-only at /autorlbench/environment. $AUTORLBENCH_WORKSPACE,
# $AUTORLBENCH_MODELS and $AUTORLBENCH_DATASETS are all writable.
#
# Assets are immutable: the model and raw dataset repos are fetched whole
# (no revision pin in the task-local fetch tool, cache-first), then the
# prepared training Parquet and frozen parity inputs are reconstructed on
# CPU and verified byte-for-byte against environment/asset_pins.json.
# Later stages read everything from inside the workspace, never host caches.
set -euo pipefail
ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${AUTORLBENCH_WORKSPACE:?The prepare stage must set AUTORLBENCH_WORKSPACE}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$AUTORLBENCH_WORKSPACE/verl${PYTHONPATH:+:$PYTHONPATH}"

# Inputs are materialized inside the workspace; later stages do not use host caches.
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
  --repo Qwen/Qwen3-VL-8B-Instruct \
  --type model \
  --workers 2 \
  --dest "$AUTORLBENCH_WORKSPACE/models/Qwen3-VL-8B-Instruct"

# Preserve the recorded model card bytes without modifying a cached hard link.
printf '%s\r\n' '---' 'license: apache-2.0' '---' > "$AUTORLBENCH_WORKSPACE/models/Qwen3-VL-8B-Instruct/.recorded-model-card"
rm -f "$AUTORLBENCH_WORKSPACE/models/Qwen3-VL-8B-Instruct/README.md"
mv "$AUTORLBENCH_WORKSPACE/models/Qwen3-VL-8B-Instruct/.recorded-model-card" "$AUTORLBENCH_WORKSPACE/models/Qwen3-VL-8B-Instruct/README.md"

python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
  --repo hiyouga/geometry3k \
  --type dataset \
  --workers 2 \
  --dest "$AUTORLBENCH_WORKSPACE/data/geometry3k_source"

# CPU-only reconstruction; independent of verl/Ray imports and local checkout paths.
python3 - --workspace "$AUTORLBENCH_WORKSPACE" --pins "$ENV_DIR/asset_pins.json" <<'PY_PREPARE_ASSETS'
#!/usr/bin/env python3
"""Prepare immutable public assets. Executed only in the network-enabled stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# The upstream CPU transformation runs separately from the framework imports.
GEO3K_PREPROCESS = r'''# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/geo3k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
'''

MODEL_REPO = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REV = "fe3340460dc1b2245a90d1b3b608bc86ffb3bfd6"
DATA_REPO = "hiyouga/geometry3k"
DATA_REV = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def create_cases(dataset, output):
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    cases = [{"name": "text_control", "images": [], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is 17 plus 25? Explain briefly."}]}]}]
    configs = [
        ("single_image_256", [0], (256, 256), False),
        ("one_turn_two_images_128", [0, 1], (128, 128), False),
        ("two_turn_two_images_128", [0, 1], (128, 128), True),
        ("one_turn_two_images_320x224", [0, 1], (320, 224), False),
        ("single_image_heldout_512x384", [2], (512, 384), False),
    ]
    for name, indices, size, two_turn in configs:
        paths = []
        for i in indices:
            image = dataset[i]["images"][0].convert("RGB")
            path = output / f"test_{i}_{size[0]}x{size[1]}.png"
            image.resize(size, Image.Resampling.BICUBIC).save(path)
            paths.append(path.name)
        if two_turn:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this geometry diagram."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "It contains labeled geometric figures."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Compare this second diagram with the first. Discuss their shapes and labels."}]},
            ]
        else:
            question = "Describe the shapes, labels, and geometric relationships in the supplied diagram(s). Explain in detail."
            messages = [{"role": "user", "content": [{"type": "image"} for _ in paths] + [{"type": "text", "text": question}]}]
        cases.append({"name": name, "images": paths, "messages": messages, "source_indices": indices,
                      "requested_size": list(size)})
    write_json(output / "cases.json", {"dataset_revision": DATA_REV, "split": "test", "cases": cases})


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    pin_path = args.pins.resolve()
    pins = json.loads(pin_path.read_text())
    expected_manifest = hashlib.sha256(pin_path.read_bytes()).hexdigest()
    raw_data = workspace / "data/geometry3k_source"
    output = workspace / "data/geo3k"
    marker = workspace / "data/.complete"
    manifest_path = workspace / "data/asset_manifest.json"
    if marker.is_file() and marker.read_text().strip() == expected_manifest and manifest_path.is_file():
        if sha256(manifest_path) == expected_manifest and all(
                (workspace / name).is_file() and (workspace / name).stat().st_size == pin["bytes"]
                and sha256(workspace / name) == pin["sha256"] for name, pin in pins["files"].items()):
            print("Pinned assets already prepared and verified.")
            return
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(workspace / "verl"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-c", GEO3K_PREPROCESS,
                    "--local_dataset_path", str(raw_data), "--local_save_dir", str(output)], check=True, env=env)
    test_data = load_dataset("parquet", data_files=str(raw_data / "data/test-00000-of-00001.parquet"), split="train")
    create_cases(test_data, workspace / "data/parity")
    for name, pin in pins["files"].items():
        path = workspace / name
        if not path.is_file() or path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
            raise RuntimeError(f"Prepared asset differs from its recorded immutable bytes: {name}")
    # Use the verified pin document verbatim, including its deterministic ordering.
    manifest_path.write_bytes(pin_path.read_bytes())
    marker.write_text(expected_manifest + "\n")
    print("Prepared and verified model, train/test data, and diagnostic inputs.")


if __name__ == "__main__":
    main()
PY_PREPARE_ASSETS
