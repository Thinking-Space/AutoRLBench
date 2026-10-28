#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned public HF converter and record input/output hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil

from checkpoint_evidence import file_sha, inspect_checkpoint, convert_bin_to_safetensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("Use a new native checkpoint directory")
    source = prepared / "pair/healthy"
    assets = json.loads((Path(__file__).parent / "assets.json").read_text())
    pair = json.loads((prepared / "pair/pair_manifest.json").read_text())
    if pair["revision"] != assets["justrl2_revision"]:
        raise RuntimeError("Prepared source revision does not match public pin")
    for name, expected in pair["healthy_files"].items():
        path = source / name
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError("Conversion source changed: " + name)
    source_inventory_sha256 = hashlib.sha256(json.dumps(pair["healthy_files"], sort_keys=True).encode()).hexdigest()
    hf_files = {}
    for asset in assets["assets"]:
        path = prepared / asset["path"]
        if path.stat().st_size != asset["size"] or file_sha(path) != asset["sha256"]:
            raise RuntimeError("Pinned HF model hash mismatch")
        hf_files[asset["path"]] = asset["sha256"]
    output.mkdir(parents=True)
    derived = output / "hf-safetensors"
    derived.mkdir()
    for asset in assets["assets"]:
        if not asset["path"].endswith(".bin"):
            shutil.copyfile(prepared / asset["path"], derived / Path(asset["path"]).name)
    tensor_check = convert_bin_to_safetensors(prepared / "model/pytorch_model.bin", derived / "model.safetensors")
    converter_input = {"format": "losslessly_reencoded_safetensors", "tensor_check": tensor_check,
                       "files": {p.name: {"size": p.stat().st_size, "sha256": file_sha(p)} for p in sorted(derived.iterdir())}}
    (output / "safetensors_receipt.json").write_text(json.dumps(converter_input, indent=2) + "\n")
    print("All public HF tensors re-encoded to safetensors and checked exactly.", flush=True)
    env = {**os.environ, "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29365",
           "PYTHONPATH": f"{source}:{source}/Megatron-LM", "HF_SOURCE": str(derived),
           "NATIVE_OUTPUT": str(output)}
    script = '''set -euo pipefail
source justrl2/model_args/minicpm5-2b.sh
python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_SOURCE" --save "$NATIVE_OUTPUT"
'''
    with (output / "conversion.log").open("w") as stream:
        subprocess.run(["bash", "-c", script], cwd=source, env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"source_revision": assets["justrl2_revision"], "model_revision": assets["model_revision"],
               "source_inventory_sha256": source_inventory_sha256,
               "converter_input": converter_input,
               "hf_files": hf_files, "native_checkpoint": inspect_checkpoint(output, expected_iteration="release"),
               "conversion_log_sha256": file_sha(output / "conversion.log")}
    (output / "conversion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Pinned public conversion completed and native payloads verified.")


if __name__ == "__main__":
    main()
