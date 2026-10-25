"""
Simple MCP Agent
Please refer to https://github.com/Snowflake-Labs/agent-world-model for more details.
"""
import asyncio
import contextlib
import io
import json
import logging
import re
import os
import shutil
import subprocess
import time
import signal
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent

from loguru import logger
from openai import AsyncOpenAI

from awm.tools import (
    tools_robust_json_loads,
    tools_jsonl_load,
    tools_json_save,
    isolated_mcp_env,
    normalize_scenario_name,
    get_random_available_port,
    async_wait_for_server,
    resolve_llm_config,
    find_scenario_entry,
)
from awm.core.server import Config as ServerConfig, _prepare_database, start_server_process


for _name in ["mcp_agent", "mcp", "httpx", "httpcore", "anyio"]:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
    logging.getLogger(_name).propagate = False
    logging.getLogger(_name).handlers = []

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings

logger.disable("mcp_agent")
logger.disable("mcp")


@dataclass
class Config:
    # Task description for the agent (optional if --scenario and --task_id are provided)
    task: str | None = None
    # Scenario name (e.g. e_commerce_33). Used with --task_id to look up tasks.
    scenario: str | None = None
    # Task index within the scenario (0-based). Used with --scenario.
    task_id: int | None = None
    # MCP server URL (streamable HTTP transport). Auto-started if --scenario is provided.
    mcp_url: str | None = None
    # OpenAI-compatible API URL (also supports Azure OpenAI via env vars)
    api_url: str | None = None
    # Model name
    model: str | None = None
    # Max agent loop iterations
    max_iterations: int = 30
    # Generation temperature
    temperature: float = 1.0
    # Max completion tokens
    max_tokens: int = 2048
    # Output directory for saving trajectory. Defaults to outputs/agents/<timestamp>
    output_dir: str | None = None
    # Path to generated environments file
    envs_path: str = "./outputs/gen_envs.jsonl"
    # Path to generated tasks file
    tasks_path: str = "./outputs/gen_tasks.jsonl"
    # Path to generated db schemas file
    db_path: str = "./outputs/gen_db.jsonl"
    # Path to generated sample data file
    sample_path: str = "./outputs/gen_sample.jsonl"
    # Verbose logging
    verbose: bool = True

    def pre_process(self):
        if self.task is None:
            print(self)
            assert self.scenario is not None and self.task_id is not None, "task is None, you must specify the scenario and task_id to lookup the task"


def get_system_prompt() -> str:
    return dedent("""\
        # MCP Tools

        You are in an MCP environment. Complete the user's task by calling MCP tools. You are already logged in; your user id is 1 when a tool requires it.

        At each step, output at most one call using exactly this format:
        <tool_call>
        {"name":"list_tools","arguments":null}
        </tool_call>

        The only top-level functions are `list_tools` and `call_tool`. Call `list_tools` exactly once at the beginning. To invoke an environment tool, use:
        <tool_call>
        {"name":"call_tool","arguments":{"tool_name":"mcp_tool_NAME","arguments":"{\\"arg\\":\\"value\\"}"}}
        </tool_call>

        The inner `arguments` value must be a valid JSON string. When the task is complete, return a plain final answer with no tool call.""")


def parse_tool_calls(content: str) -> list[dict]:
    tool_calls = []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        data = tools_robust_json_loads(match.strip())
        if not data:
            logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
            continue

        # Handle list wrapping: [{"name": ..., "arguments": ...}]
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                continue

        if not isinstance(data, dict):
            continue

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        if name.startswith("mcp_tool_"):
            arguments = {
                "tool_name": name,
                "arguments": arguments if arguments else {},
            }
            name = "call_tool"

        tool_calls.append({
            "id": f"call_{int(time.time() * 1000)}_{i}",
            "name": name,
            "arguments": arguments,
        })

    return tool_calls


def parse_call_tool_arguments(arguments: dict | str | None) -> tuple[str, dict]:
    if isinstance(arguments, str):
        arguments = tools_robust_json_loads(arguments)
    if not isinstance(arguments, dict):
        return "", {}

    tool_name = arguments.get("tool_name", "")
    inner_args = arguments.get("arguments", {})

    if tool_name.startswith("mcp_tool_"):
        tool_name = tool_name[len("mcp_tool_"):]

    if isinstance(inner_args, str):
        parsed = tools_robust_json_loads(inner_args) if inner_args.strip() else {}
        if isinstance(parsed, dict):
            inner_args = parsed
        else:
            inner_args = {}

    if not isinstance(inner_args, dict):
        inner_args = {}

    return tool_name, inner_args


def format_tools_for_response(tools: list[dict]) -> str:

    def format_input_schema(schema: dict, indent_level: int = 6, parent_required: list | None = None) -> str:
        if not schema:
            return ""

        result = ""
        indent_str = " " * indent_level
        properties = schema.get('properties', {})
        required_fields = parent_required if parent_required is not None else schema.get('required', [])

        for prop_name, prop_info in properties.items():
            is_required = prop_name in required_fields
            required_str = " (required)" if is_required else " (optional)"
            prop_type = prop_info.get('type', 'unknown')
            description = prop_info.get('description', '')
            default = prop_info.get('default')
            enum_values = prop_info.get('enum')
            nested_properties = prop_info.get('properties', {})
            nested_required = prop_info.get('required', [])

            result += f"{indent_str}- {prop_name}: {prop_type}{required_str}\n"
            if description:
                result += f"{indent_str}  Description: {description}\n"
            if default is not None:
                result += f"{indent_str}  Default: {default}\n"
            if enum_values:
                result += f"{indent_str}  Allowed values: {enum_values}\n"

            if prop_type == "object" and nested_properties:
                result += f"{indent_str}  Properties:\n"
                nested_schema = {"properties": nested_properties, "required": nested_required}
                result += format_input_schema(nested_schema, indent_level + 4, nested_required)

        return result

    # filter out list_tools meta-tool
    actual_tools = [t for t in tools if t.get('name') != 'list_tools']

    docs_text = f"Available MCP Tools ({len(actual_tools)} tools):\n"
    docs_text += "=" * 80 + "\n\n"

    for i, tool in enumerate(actual_tools, 1):
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", tool.get("input_schema", {}))

        # add mcp_tool_ prefix
        mcp_name = f"mcp_tool_{name}" if not name.startswith("mcp_tool_") else name

        # parse description for multi-line
        desc_lines = description.split('\n')
        first_line = desc_lines[0].strip() if desc_lines else "No description"
        more_desc = '\n'.join(line.strip() for line in desc_lines[1:]).strip() if len(desc_lines) > 1 else ""

        docs_text += f"{i}. {mcp_name}\n"
        docs_text += f"   Description: {first_line}\n"
        if more_desc:
            for line in more_desc.split('\n'):
                if line.strip():
                    docs_text += f"   {line}\n"

        if input_schema and input_schema.get('properties'):
            docs_text += f"   Parameters:\n"
            docs_text += format_input_schema(input_schema)
        else:
            docs_text += f"   Parameters: None\n"

        docs_text += "\n"

    return docs_text.strip()


MCP_SERVER_NAME = "mcp_server"


class MCPToolExecutor:

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict] = []

        with isolated_mcp_env():
            settings = Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(
                    type="none",
                    transports=["none"],
                    progress_display=False,
                    level="error",
                ),
                mcp=MCPSettings(
                    servers={
                        MCP_SERVER_NAME: MCPServerSettings(
                            transport='streamable_http',
                            url=self.mcp_url,
                        ),
                    }
                ),
            )
            self._app = MCPApp(name="demo_agent", settings=settings)
            self._agent = Agent(name="executor", server_names=[MCP_SERVER_NAME])

    async def _run_mcp(self, operation):
        # Cover MCPApp/Agent enter/exit as well as the tool RPC.  Do not use
        # wait_for() here: it waits for cancellation to finish, so a hung
        # streamable-HTTP __aexit__ still stalls the whole rollout gather.
        async def _run():
            with contextlib.redirect_stderr(io.StringIO()):
                async with self._app.run():
                    async with self._agent:
                        return await operation()
        task = asyncio.create_task(_run())
        done, _pending = await asyncio.wait({task}, timeout=self.timeout)
        if task in done:
            return task.result()
        task.cancel()
        raise asyncio.TimeoutError(f"MCP operation timed out after {self.timeout}s")

    async def list_tools(self) -> list[dict]:
        last_tools: list[dict] = []
        for attempt in range(3):
            async def _list():
                result = await self._agent.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema or {},
                    }
                    for t in result.tools
                ]
            try:
                last_tools = await self._run_mcp(_list)
            except asyncio.TimeoutError:
                last_tools = []
            if last_tools or attempt == 2:
                self._tools = last_tools
                return last_tools
            await asyncio.sleep(1.0 * (attempt + 1))
        return last_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        async def _call():
            result = await self._agent.call_tool(tool_name, arguments)
            parts = []
            for c in result.content:
                if hasattr(c, 'text'):
                    parts.append(c.text)
                else:
                    parts.append(str(c))
            text = "\n".join(parts)
            if result.isError:
                return f"Error: {text}"
            return text
        return await self._run_mcp(_call)


async def generate_response(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    config: Config,
    use_vllm_extras: bool = False,
) -> tuple[str, list[dict]]:
    api_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role in ("system", "user"):
            api_messages.append({"role": role, "content": content})
        elif role == "assistant":
            api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            if use_vllm_extras:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", "call_unknown"),
                    "content": content,
                })
            else:
                # For standard OpenAI API, tool role requires tool_calls in the assistant message.
                # Since we use XML-based tool calls, send tool responses as user messages.
                api_messages.append({"role": "user", "content": f"Tool response:\n{content}"})

    kwargs = dict(
        model=model_name,
        messages=api_messages,
        max_completion_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    if use_vllm_extras:
        kwargs["extra_body"] = {
            "add_generation_prompt": True,
            "min_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    last_error = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            logger.warning(f"LLM request failed (attempt {attempt + 1}/3): {exc}; retrying")
            await asyncio.sleep(1.5 * (attempt + 1))
    else:
        raise last_error

    content = response.choices[0].message.content or ""
    tool_calls = parse_tool_calls(content)

    return content, tool_calls



async def run_agent(config: Config):
    # Resolve scenario/task
    scenario = config.scenario
    task_id = config.task_id

    if config.task:
        task = config.task
    
    elif config.scenario and config.task_id is not None:
        tasks_data = tools_jsonl_load(config.tasks_path)
        entry = find_scenario_entry(tasks_data, scenario)
        if not entry:
            raise ValueError(f"Scenario '{scenario}' not found in {config.tasks_path}")
        tasks = entry["tasks"]
        if task_id < 0 or task_id >= len(tasks):
            raise ValueError(f"task_id {task_id} out of range (0-{len(tasks)-1}) for scenario {scenario}")
        task = tasks[task_id]
        logger.info(f"Resolved task from scenario={scenario}, task_id={task_id}: {task}")

    else:
        raise ValueError("Either --task or --scenario (with optional --task_id) must be provided")

    # Resolve LLM config
    api_url, api_key, model = resolve_llm_config(
        api_url_override=config.api_url,
        model_override=config.model,
    )
    use_vllm_extras = "localhost" in api_url and "openai.azure.com" not in api_url and "v1" in api_url
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir:
        output_dir = config.output_dir
    else:
        run_name = f"{timestamp}"
        if scenario:
            run_name = f"{timestamp}_{scenario}_task_{task_id}"
        output_dir = os.path.join("outputs", "agents", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup database and MCP server if scenario is provided
    server_proc = None
    db_file_path = None

    if scenario and not config.mcp_url:
        # Prepare database (reuse server module logic)
        server_cfg = ServerConfig(
            scenario=scenario,
            envs_load_path=config.envs_path,
            db_schema_path=config.db_path,
            sample_path=config.sample_path,
            temp_server_path=os.path.join(output_dir, "server_runtime.py"),
        )
        db_file_path = _prepare_database(server_cfg, output_dir)

        # Start MCP server
        port = get_random_available_port()
        server_proc = start_server_process(
            scenario,
            config.envs_path,
            db_file_path,
            port,
            output_dir=output_dir,
            temp_server_path=server_cfg.temp_server_path,
        )
        mcp_url = f"http://127.0.0.1:{port}/mcp"

        logger.info(f"Waiting for MCP server on port {port}...")
        if not await async_wait_for_server(port, timeout=60):
            if server_proc.poll() is not None:
                stderr = server_proc.stderr.read().decode() if server_proc.stderr else ""
                logger.error(f"MCP server process exited with code {server_proc.returncode}. stderr: {stderr[:1000]}")
            raise RuntimeError(f"MCP server failed to start on port {port}")
        logger.info(f"MCP server ready at {mcp_url}")
    else:
        mcp_url = config.mcp_url
        if not mcp_url:
            raise ValueError("--mcp_url is required when --scenario is not provided")

    try:
        logger.info("=" * 80)
        logger.info("MCP Agent")
        logger.info("=" * 80)
        logger.info(f"Task: {task}")
        logger.info(f"MCP Server: {mcp_url}")
        logger.info(f"LLM API: {api_url}")
        logger.info(f"Model: {model}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)

        mcp = MCPToolExecutor(mcp_url)
        llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_url,
        )

        logger.info("Connecting to MCP server and fetching tools...")
        tools = await mcp.list_tools()
        logger.info(f"Loaded {len(tools)} tools from MCP server")
        tools_response_text = format_tools_for_response(tools)

        messages: list[dict] = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": task},
        ]

        # Trajectory recording
        trajectory: list[dict] = []

        iteration = 0
        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{config.max_iterations} ---")

            content, tool_calls = await generate_response(
                llm_client, model, messages, config, use_vllm_extras=use_vllm_extras,
            )

            if config.verbose:
                preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"Assistant ({len(content)} chars): {preview}")
                logger.info(f"Tool calls: {len(tool_calls)}")

            messages.append({"role": "assistant", "content": content})

            # no tool calls -> task complete
            if not tool_calls:
                logger.info("No tool calls detected - task complete.")
                logger.info(f"\n{'=' * 40} Final Answer {'=' * 40}")
                logger.info(content)
                logger.info("=" * 80)
                trajectory.append({
                    "iteration": iteration,
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [],
                    "is_final": True,
                })
                break

            # process first tool call only (one per turn)
            tc = tool_calls[0]
            name = tc["name"]
            arguments = tc["arguments"]
            tool_call_id = tc["id"]

            if len(tool_calls) > 1:
                skipped = [t["name"] for t in tool_calls[1:]]
                logger.warning(f"Multiple tool calls detected. Only executing first: {name}. Skipped: {skipped}")

            # execute tool call
            if name == "list_tools":
                logger.info("Executing: list_tools")
                response_text = tools_response_text

            elif name == "call_tool":
                tool_name, tool_args = parse_call_tool_arguments(arguments)
                logger.info(f"Executing: call_tool({tool_name}, {json.dumps(tool_args, ensure_ascii=False)[:200]})")
                try:
                    response_text = await mcp.call_tool(tool_name, tool_args)
                except asyncio.TimeoutError:
                    response_text = f"Error: Tool call timed out after {mcp.timeout}s"
                    logger.error(response_text)
                except Exception as e:
                    response_text = f"Error: {e}"
                    logger.error(f"Tool call failed: {e}")
            else:
                response_text = f"Error: Unknown tool '{name}'. Only 'list_tools' and 'call_tool' are available."

            if config.verbose:
                preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logger.info(f"Tool response ({len(response_text)} chars): {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": response_text,
            })

            # Record trajectory entry
            trajectory.append({
                "iteration": iteration,
                "role": "assistant",
                "content": content,
                "tool_calls": [tc],
                "tool_response": {
                    "tool_call_id": tool_call_id,
                    "content": response_text,
                },
            })
        else:
            logger.warning("Max iterations reached without completion.")

        logger.info("=" * 80)
        logger.info(f"Agent execution complete. Total iterations: {iteration}")
        logger.info("=" * 80)

        # Save trajectory and outputs
        trajectory_data = {
            "scenario": scenario,
            "task_id": task_id,
            "task": task,
            "model": model,
            "api_url": api_url,
            "max_iterations": config.max_iterations,
            "temperature": config.temperature,
            "total_iterations": iteration,
            "timestamp": timestamp,
            "trajectory": trajectory,
            "messages": messages,
        }
        trajectory_path = os.path.join(output_dir, "trajectory.json")
        tools_json_save(trajectory_data, trajectory_path)
        logger.info(f"Saved trajectory to {trajectory_path}")

        # Save final database
        if db_file_path and os.path.exists(db_file_path):
            final_db_path = os.path.join(output_dir, "final.db")
            if os.path.abspath(db_file_path) != os.path.abspath(final_db_path):
                shutil.copy2(db_file_path, final_db_path)
            logger.info(f"Saved final database to {final_db_path}")

        logger.info(f"Run outputs saved to: {output_dir}")

    finally:
        # Cleanup MCP server
        if server_proc and server_proc.poll() is None:
            logger.info("Shutting down MCP server...")
            try:
                os.killpg(server_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server_proc.wait()


def run(config: Config):
    asyncio.run(run_agent(config))


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)
