#!/usr/bin/env python3
"""Run vLLM 0.11 with the Transformers 5 tokenizer API used by IDEA."""

import os
from pathlib import Path


def _install_transformers_compatibility() -> None:
    # sitecustomize is imported in every multiprocessing child as well as the
    # API server, so the vLLM engine core sees the same compatibility shim.
    shim_dir = str(Path(__file__).resolve().parent)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if shim_dir not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(
            part for part in (shim_dir, pythonpath) if part
        )
    # Apply it in this API-server process too; sitecustomize handles children.
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda tokenizer: tokenizer.all_special_tokens
        )


if __name__ == "__main__":
    _install_transformers_compatibility()

    from vllm.entrypoints.cli.main import main

    main()
