#!/usr/bin/env python3
"""Reproduce and inspect the first two AWM agent turns for appointment_scheduling_1."""





from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

AWM_ROOT = Path("/workspace/awm_source")
DATA_ROOT = Path("/workspace/data/AgentWorldModel-1K")
sys.path.insert(0, str(AWM_ROOT))

from openai import AsyncOpenAI

from awm.core.agent import Config, MCPToolExecutor, get_system_prompt, parse_tool_calls
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
    raise LookupError(f"{SCENARIO} was not found in {DATA_ROOT / 'gen_tasks.jsonl'}")


def remove_think_content(value: str) -> str:
    """Remove private reasoning from a model response before reusing it as history."""
    return re.sub(r"<think>.*?</think>\s*", "", value, flags=re.IGNORECASE | re.DOTALL)


def api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror awm.core.agent.generate_response's vLLM message conversion."""
    return [
        {
            "role": item["role"],
            "content": remove_think_content(item.get("content", "")) if item["role"] == "assistant" else item.get("content", ""),
            **({"tool_call_id": item["tool_call_id"]} if item["role"] == "tool" else {}),
        }
        for item in messages
    ]


def excerpt(value: str, limit: int = 1800) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    return f"{value[:half]}\n\n... <{len(value) - 2 * half} chars omitted> ...\n\n{value[-half:]}"


def print_request(turn: int, messages: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 30} TURN {turn}: REQUEST {'=' * 30}")
    for index, message in enumerate(messages, start=1):
        content = message.get("content", "")
        print(f"[{index}] role={message['role']} chars={len(content)}")
        print(excerpt(content))


def print_response(turn: int, content: str, calls: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 30} TURN {turn}: RESPONSE {'=' * 29}")
    print(f"chars={len(content)} parsed_tool_calls={len(calls)}")
    print(excerpt(content))
    print("\nparsed calls:")
    print(json.dumps(calls, ensure_ascii=False, indent=2))


async def request(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    config: Config,
    enable_thinking: bool,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=api_messages(messages),
        max_completion_tokens=config.max_tokens,
        temperature=config.temperature,
        extra_body={
            "add_generation_prompt": True,
            "min_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )
    return response.choices[0].message.content or ""


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
        print(f"MCP server: http://127.0.0.1:{port}/mcp")

        api_url, api_key, model = resolve_llm_config(args.api_url, args.model)
        config = Config(max_tokens=args.max_tokens, temperature=args.temperature)
        client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        mcp = MCPToolExecutor(f"http://127.0.0.1:{port}/mcp")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": task},
        ]

        print_request(1, api_messages(messages))
        first = await request(client, model, messages, config, args.enable_thinking)
        first_calls = parse_tool_calls(first)
        print_response(1, first, first_calls)
        if not first_calls or first_calls[0]["name"] != "list_tools":
            raise RuntimeError("Turn 1 did not produce the expected list_tools call")

        tools = await mcp.list_tools()
        # This function is deliberately imported only after successful MCP setup.
        from awm.core.agent import format_tools_for_response
        tool_result = format_tools_for_response(tools)
        messages.extend([
            {"role": "assistant", "content": first},
            {"role": "tool", "tool_call_id": first_calls[0]["id"], "content": tool_result},
        ])

        second_request = api_messages(messages)
        print_request(2, second_request)
        second = await request(client, model, messages, config, args.enable_thinking)
        second_calls = parse_tool_calls(second)
        print_response(2, second, second_calls)

        artifact = {
            "created_at": datetime.now().isoformat(), "model": model, "api_url": api_url,
            "mcp_url": f"http://127.0.0.1:{port}/mcp", "task": task,
            "turn_1": {"request": api_messages(messages[:2]), "response": first, "calls": first_calls},
            "turn_2": {"request": second_request, "response": second, "calls": second_calls},
        }
        (output_dir / "turns.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nFull, untruncated transcript: {output_dir / 'turns.json'}")
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
    parser.add_argument("--model", default="qwen3-8b-awm-natural-step-715")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking in the chat template (default: disabled).",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/idea_source/outputs/awm_debug_appointment_second_turn",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
