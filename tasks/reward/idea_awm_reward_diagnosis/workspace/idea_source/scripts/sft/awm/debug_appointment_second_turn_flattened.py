#!/usr/bin/env python3
"""Test appointment turns using the flattened prompt format used by SFT data."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import sys

AWM_ROOT = Path("/workspace/awm_source")
DATA_ROOT = Path("/workspace/data/AgentWorldModel-1K")
sys.path.insert(0, str(AWM_ROOT))

from awm.core.agent import Config, MCPToolExecutor, format_tools_for_response, get_system_prompt, parse_tool_calls
from awm.core.server import Config as ServerConfig, _prepare_database, start_server_process
from awm.tools import async_wait_for_server, get_random_available_port, resolve_llm_config


SCENARIO = "appointment_scheduling_1"
TASK_INDEX = 3


def load_task() -> str:
    with (DATA_ROOT / "gen_tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("scenario") == SCENARIO:
                return str(row["tasks"][TASK_INDEX])
    raise LookupError(f"{SCENARIO} was not found")


def remove_think(value: str) -> str:
    value = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in value:
        value = value.split("</think>", 1)[1]
    return value.strip()


def flattened_prompt(history: list[dict[str, Any]]) -> str:
    """Match build_candidate_skill_sft.py/merge_turn_sft_data.py serialization."""
    parts = [
        "You are completing a multi-turn tool-use task. Continue the conversation "
        "and produce the next assistant action or final answer."
    ]
    for message in history:
        role = str(message.get("role", "user"))
        content = remove_think(str(message.get("content", "")))
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


async def request(client: AsyncOpenAI, model: str, prompt: str, config: Config) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=config.max_tokens,
        temperature=config.temperature,
        extra_body={
            "add_generation_prompt": True,
            "min_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content or ""


def show(turn: int, prompt: str, response: str) -> None:
    print(f"\n{'=' * 28} TURN {turn} {'=' * 28}")
    print(f"prompt_chars={len(prompt)} response_chars={len(response)}")
    print(response[:4000])
    print("\nparsed_tool_calls:")
    print(json.dumps(parse_tool_calls(response), ensure_ascii=False, indent=2))


async def main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_proc = None
    try:
        task = load_task()
        server_config = ServerConfig(
            scenario=SCENARIO,
            envs_load_path=str(DATA_ROOT / "gen_envs.jsonl"),
            db_schema_path=str(DATA_ROOT / "gen_db.jsonl"),
            sample_path=str(DATA_ROOT / "gen_sample.jsonl"),
            output_dir=str(output_dir),
            temp_server_path=str(output_dir / "server_runtime.py"),
        )
        database = _prepare_database(server_config, str(output_dir))
        port = get_random_available_port()
        server_proc = start_server_process(
            SCENARIO, str(DATA_ROOT / "gen_envs.jsonl"), database, port,
            output_dir=str(output_dir), temp_server_path=server_config.temp_server_path,
        )
        if not await async_wait_for_server(port, timeout=60):
            raise RuntimeError(f"MCP server did not start on port {port}")

        api_url, api_key, model = resolve_llm_config(args.api_url, args.model)
        client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        config = Config(max_tokens=args.max_tokens, temperature=args.temperature)
        mcp = MCPToolExecutor(f"http://127.0.0.1:{port}/mcp")

        history = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": task},
        ]
        prompt_1 = flattened_prompt(history)
        first = await request(client, model, prompt_1, config)
        show(1, prompt_1, first)
        first_calls = parse_tool_calls(first)
        if not first_calls or first_calls[0]["name"] != "list_tools":
            raise RuntimeError("Turn 1 did not produce the expected list_tools call")

        tools = await mcp.list_tools()
        tool_result = format_tools_for_response(tools)
        history.extend([
            {"role": "assistant", "content": first},
            {"role": "tool", "tool_call_id": first_calls[0]["id"], "content": tool_result},
        ])
        prompt_2 = flattened_prompt(history)
        second = await request(client, model, prompt_2, config)
        show(2, prompt_2, second)

        artifact = {
            "created_at": datetime.now().isoformat(),
            "format": "flattened_sft_prompt",
            "model": model,
            "api_url": api_url,
            "task": task,
            "turn_1": {"prompt": prompt_1, "response": first, "calls": first_calls},
            "turn_2": {"prompt": prompt_2, "response": second, "calls": parse_tool_calls(second)},
        }
        (output_dir / "turns_flattened.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nTranscript: {output_dir / 'turns_flattened.json'}")
    finally:
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except Exception:
                server_proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8220/v1")
    parser.add_argument("--model", default="global_step_100")
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="/workspace/idea_source/outputs/awm_debug_appointment_second_turn_flattened",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
