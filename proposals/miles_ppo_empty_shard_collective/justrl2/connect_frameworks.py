#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Link frameworks from the pinned candidate image into both prepared workspaces."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    root = args.prepared.resolve()
    framework_paths = {"Megatron-LM": Path("/root/Megatron-LM"), "sglang": Path("/sgl-workspace/sglang")}
    receipt = {"frameworks": {}, "packages": {}}
    for name, source in framework_paths.items():
        if not source.is_dir():
            raise RuntimeError(f"Candidate image missing {source}")
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        receipt["frameworks"][name] = {"revision": revision}
        for kind in ["healthy", "faulty"]:
            target = root / "pair" / kind / name
            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f"Unexpected existing framework at {target}")
            else:
                target.symlink_to(source, target_is_directory=True)
    for name in ["torch", "transformers", "numpy", "ray", "sglang", "transformer-engine", "mbridge"]:
        receipt["packages"][name] = importlib.metadata.version(name)
    (root / "framework_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
