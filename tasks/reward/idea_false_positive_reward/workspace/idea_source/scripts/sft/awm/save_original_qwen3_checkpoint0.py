#!/usr/bin/env python3
"""Save the unmodified Qwen3-8B model using the VERL SFT checkpoint format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local/user/Qwen/Qwen3-8B")
    parser.add_argument(
        "--output",
        default="/workspace/idea_source/outputs/qwen3_8b_awm_skill_sft_verl_structured/global_step_0",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # The original model is already in Hugging Face sharded safetensors form.
    # Copying those immutable shards avoids materializing a second 16 GB state
    # dict, while producing the exact files consumed by from_pretrained/vLLM.
    source = Path(args.model)
    names = [
        "config.json", "configuration.json", "generation_config.json",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json",
    ]
    names += [p.name for p in source.glob("model-*-of-*.safetensors")]
    for name in names:
        src = source / name
        if src.is_file():
            dst = output / name
            if dst.exists():
                dst.unlink()
            # Same NFS filesystem: hard links are instantaneous and preserve
            # the exact original bytes without a second 16 GB allocation.
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    print(f"Saved original model and tokenizer to {output}")


if __name__ == "__main__":
    main()
