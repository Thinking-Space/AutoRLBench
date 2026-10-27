#!/usr/bin/env python3
"""Probe how the AWM MCP server reports common tool-call failures.

This is an integration test: it starts a disposable scenario server and calls
the MCP endpoint directly, without involving an LLM.  The printed result is
the exact text returned by ``MCPToolExecutor.call_tool``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

AWM_ROOT = Path("/workspace/awm_source")
DATA_ROOT = Path("/workspace/data/AgentWorldModel-1K")
sys.path.insert(0, str(AWM_ROOT))

from awm.core.agent import MCPToolExecutor  # noqa: E402
from awm.core.server import Config as ServerConfig, _prepare_database, start_server_process  # noqa: E402
from awm.tools import async_wait_for_server, get_random_available_port  # noqa: E402


def required_parameters(tool: dict[str, Any]) -> list[str]:
    schema = tool.get("inputSchema") or {}
    return [str(name) for name in (schema.get("required") or [])]


async def main_async(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = args.scenario
    server_config = ServerConfig(
        scenario=scenario,
        envs_load_path=str(DATA_ROOT / "gen_envs.jsonl"),
        db_schema_path=str(DATA_ROOT / "gen_db.jsonl"),
        sample_path=str(DATA_ROOT / "gen_sample.jsonl"),
        output_dir=str(output_dir),
        temp_server_path=str(output_dir / "server_runtime.py"),
    )
    database = _prepare_database(server_config, str(output_dir))
    port = get_random_available_port()
    process = start_server_process(
        scenario,
        str(DATA_ROOT / "gen_envs.jsonl"),
        database,
        port,
        output_dir=str(output_dir),
        temp_server_path=server_config.temp_server_path,
    )
    rows: list[dict[str, Any]] = []
    try:
        if not await async_wait_for_server(port, timeout=60):
            raise RuntimeError(f"MCP server did not start on port {port}")
        executor = MCPToolExecutor(f"http://127.0.0.1:{port}/mcp")
        tools = await executor.list_tools()
        if not tools:
            raise RuntimeError("MCP server returned no tools")

        # Pick a known tool with required args for deterministic validation probes.
        target = next((t for t in tools if required_parameters(t)), tools[0])
        required = required_parameters(target)
        probes = [
            ("unknown_tool", "__definitely_not_a_real_tool__", {}),
            ("missing_required_arguments", target["name"], {}),
        ]
        if required:
            probes.append(("wrong_argument_type", target["name"], {required[0]: {"invalid": True}}))
        for case, tool_name, arguments in probes:
            try:
                result = await executor.call_tool(tool_name, arguments)
                row = {"case": case, "tool": tool_name, "arguments": arguments,
                       "returned": result, "is_error_prefix": result.startswith("Error:")}
            except Exception as exc:
                row = {"case": case, "tool": tool_name, "arguments": arguments,
                       "raised": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))

        output = output_dir / "tool_error_returns.json"
        output.write_text(json.dumps({"scenario": scenario, "port": port, "tools": tools,
                                      "probes": rows}, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"\nFull results: {output}")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="appointment_scheduling_1")
    parser.add_argument("--output-dir", default="outputs/awm_mcp_tool_error_returns")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
