#!/usr/bin/env python3
"""Send one real AWM SFT validation example to the step-300 vLLM server."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd
from openai import OpenAI


DEFAULT_DATA = Path(
    "/workspace/idea_source/outputs/awm_qwen38_sft_data/sft_episode_skill_val.parquet"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200/v1")
    parser.add_argument("--model", default="qwen3-8b-step-300")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    rows = pd.read_parquet(args.data)
    if not 0 <= args.row < len(rows):
        raise SystemExit(f"--row must be in [0, {len(rows)})")
    prompt = str(rows.iloc[args.row]["prompt"])

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=600.0,
        http_client=httpx.Client(trust_env=False, timeout=600.0),
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.perf_counter() - started
    output = response.choices[0].message.content or ""

    print("=== input ===")
    print(prompt)
    print("=== output ===")
    print(output)
    print("=== timing ===")
    print(f"response_time_sec={elapsed:.3f}")
    print(f"input_tokens={getattr(response.usage, 'prompt_tokens', None)}")
    print(f"output_tokens={getattr(response.usage, 'completion_tokens', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
