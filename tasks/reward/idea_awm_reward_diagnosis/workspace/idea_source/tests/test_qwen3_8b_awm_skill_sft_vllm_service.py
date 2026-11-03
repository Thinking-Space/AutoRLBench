#!/usr/bin/env python3
"""Smoke-test an AWM SFT checkpoint through an OpenAI-compatible vLLM service.

The first two prompts are ordinary user questions.  The third prompt is read
from the same parquet format used by ``run_qwen3_8b_verl_sft.sh`` and is sent
unchanged to the vLLM completions endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from openai import OpenAI


DEFAULT_CHECKPOINT_ROOT = Path("/local/user/models/qwen3_8b_awm_skill_sft_verl_12288")
DEFAULT_DATA = Path("/workspace/idea_source/outputs/awm_qwen38_sft_data/sft_episode_skill_train.parquet")


def read_training_prompt(data_path: Path, row: int) -> str:
    table = pq.read_table(data_path, columns=["prompt"])
    if row < 0 or row >= table.num_rows:
        raise ValueError(f"--row must be in [0, {table.num_rows})")
    value = table.column("prompt")[row].as_py()
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training row {row} has an empty prompt")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=700, help="checkpoint global step")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--row", type=int, default=0, help="training parquet row for the third input")
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--model", default=None, help="served model name (defaults to qwen3-8b-awm-sft-step-N)")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    checkpoint = args.checkpoint_root / f"global_step_{args.step}"
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    model_name = args.model or f"qwen3-8b-awm-sft-step-{args.step}"
    prompts = [
        ("common-1", "Explain in one short paragraph why unit tests are useful."),
        ("common-2", "What is the capital of France? Answer concisely."),
        ("training", read_training_prompt(args.data, args.row)),
    ]

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=600.0,
        http_client=httpx.Client(trust_env=False, timeout=600.0),
    )
    try:
        served_models = [item.id for item in client.models.list().data]
        if model_name not in served_models:
            raise RuntimeError(f"served models are {served_models}")
    except Exception as exc:
        raise SystemExit(
            f"vLLM service is unavailable or does not serve {model_name!r} at {args.base_url}: {exc}"
        ) from exc

    outputs = []
    for label, prompt in prompts:
        # SFTDataset wraps every parquet prompt in the model chat template and
        # appends an assistant generation marker.  Use chat completions here as
        # well, including for the training-shaped prompt, so inference matches
        # the format used during training.
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = response.choices[0].message.content or ""
        print(f"=== {label} input ===")
        print(prompt)
        print(f"=== {label} output ===")
        print(output)
        if not output.strip():
            raise AssertionError(f"empty output for {label}")
        outputs.append({"label": label, "input": prompt, "output": output})

    print(json.dumps({"step": args.step, "model": model_name, "samples": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
