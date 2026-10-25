import re
import os
import orjson
import json_repair
import json
import time
import socket
import subprocess
import asyncio
import contextlib
import io
import random

from tiktoken import encoding_for_model
from json_repair import repair_json
from pathlib import Path
from loguru import logger
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings


def dump_sqlite_to_string(db_path: str) -> str:
    from contextlib import closing
    import sqlite3
    
    with closing(sqlite3.connect(db_path)) as conn:
        conn.text_factory = str

        try:
            return "\n".join(conn.iterdump())
        except sqlite3.Error as e:
            logger.error(f"{db_path} [dump_sqlite_to_string] iterdump failed: {e!r}, fallback to manual dump.")

        lines: list[str] = []
        cur = conn.cursor()

        lines.append("BEGIN TRANSACTION;")

        # manual dump
        for name, sql in cur.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY name
            """
        ):
            lines.append(f"{sql};")

            try:
                data_cur = conn.execute(f'SELECT * FROM "{name}"')
                col_count = len(data_cur.description)

                for row in data_cur:
                    values_sql = []
                    for v in row:
                        if v is None:
                            values_sql.append("NULL")
                        elif isinstance(v, (int, float)):
                            values_sql.append(str(v))
                        else:
                            q = conn.execute("SELECT quote(?)", (v,)).fetchone()[0]
                            values_sql.append(q)

                    values_str = ",".join(values_sql)
                    lines.append(f'INSERT INTO "{name}" VALUES({values_str});')

            except sqlite3.Error as e:
                lines.append(f"-- error dumping data for table {name}: {e!r}")

        for type_ in ("index", "trigger", "view"):
            for name, sql in cur.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = ?
                  AND name NOT LIKE 'sqlite_%'
                  AND sql IS NOT NULL
                ORDER BY name
                """,
                (type_,),
            ):
                lines.append(f"{sql};")

        lines.append("COMMIT;")
        dump_str = "\n".join(lines)

        logger.debug(f"{db_path} [dump_sqlite_to_string] manual dump done, {len(lines)} lines.\n{dump_str[:1000]}")

        return dump_str


def get_random_available_port(start_port: int | None = None, end_port: int | None = None, max_attempts: int = 100) -> int:
    
    if start_port is None or end_port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    for _ in range(max_attempts):
        port = random.randint(start_port, end_port - 1)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                s.listen(1)
            return port
        except OSError:
            continue
    
    raise RuntimeError(f"no available port found in range {start_port}-{end_port} after {max_attempts} attempts")

@contextlib.contextmanager
def isolated_mcp_env():
    MCP_ENV_SAFE_PREFIXES = ['HOME', 'USER', 'PATH', 'LANG', 'TERM', 'SHELL', 'PWD', 'TMPDIR', 'TMP', 'TEMP',]
    MCP_ENV_SAFE_KEYS = ['PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV']

    saved_env = dict(os.environ)
    
    keys_to_remove = []
    for key in os.environ.keys():
        is_safe = any(key.startswith(prefix) for prefix in MCP_ENV_SAFE_PREFIXES) or key in MCP_ENV_SAFE_KEYS
        if not is_safe:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del os.environ[key]
    
    try:
        yield
    finally:
        # restore all environment variables
        os.environ.clear()
        os.environ.update(saved_env)

async def check_mcp_server(url: str, timeout: float = 10.0) -> tuple[bool, int, list[dict], str | None]:
    """
    return (is_running, tools_count, tools, error)
    """

    MCP_SERVER_NAME = "mcp_tool"

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
                        url=url
                    ),
                }
            ),
        )
        
        app = MCPApp(name="_mcp_probe", settings=settings)
        
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                async with app.run():
                    agent = Agent(name="_probe", server_names=[MCP_SERVER_NAME])
                    async with agent:
                        tools_result = await asyncio.wait_for(agent.list_tools(), timeout=timeout)
                        tools = []
                        for t in tools_result.tools:
                            tool_info = {
                                "name": t.name,
                                "description": t.description,
                                "inputSchema": t.inputSchema,
                            }
                            if t.title:
                                tool_info["title"] = t.title
                            if t.outputSchema:
                                tool_info["outputSchema"] = t.outputSchema
                            if t.annotations:
                                tool_info["annotations"] = t.annotations.model_dump_json()
                            if t.meta:
                                tool_info["meta"] = t.meta
                            
                            tools.append(tool_info)
                        
                        if len(tools) == 0:
                            return (False, 0, [], "No tools available from MCP server")
                        return (True, len(tools), tools, None)
            except asyncio.TimeoutError:
                return (False, 0, [], f"Timeout after {timeout}s")
            except Exception as e:
                return (False, 0, [], str(e))

async def async_wait_for_server(port: int, timeout: float = 60.0) -> bool:
    """Wait for an MCP server to become available on the given port."""
    start_time = time.time()
    last_err = None
    while time.time() - start_time < timeout:
        try:
            running, tools_count, tools, err = await check_mcp_server(
                url=f"http://localhost:{port}/mcp", timeout=10
            )
            if running and tools_count and len(tools) > 0:
                return True
            else:
                last_err = err
        except Exception:
            pass
        await asyncio.sleep(0.5)

    if last_err:
        logger.error(f"Failed to connect to MCP server after {timeout}s: {last_err}")
    return False


def wait_for_server(port: int, timeout: float = 60.0) -> bool:
    """Sync wrapper around async_wait_for_server."""
    return asyncio.run(async_wait_for_server(port, timeout=timeout))

def kill_process_on_port(port):
    try:
        result = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pids = result.decode().strip().split()
        for pid in pids:
            print(f"Killing PID {pid} on port {port}...")
            os.system(f"kill -9 {pid}")
    except subprocess.CalledProcessError:
        print(f"No process found on port {port}.")

def is_port_available(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('localhost', port))
        time.sleep(0.1)
        return True
    except OSError:
        return False

def wait_port_free(port, timeout=10) -> bool:
    if timeout < 1:
        return False
    start = time.time()
    while time.time() - start < timeout:
        if is_port_available(port):
            return True
        time.sleep(0.2)
    kill_process_on_port(port)
    return wait_port_free(port, timeout // 2)

def _sanitize_for_json_utf8(obj):
    if isinstance(obj, dict):
        return {_sanitize_for_json_utf8(k): _sanitize_for_json_utf8(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json_utf8(item) for item in obj]
    elif isinstance(obj, str):
        return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_json_utf8(item) for item in obj)
    else:
        return obj


def format_db_schema(db_schema: dict) -> str:
    parts = ["Database Schema (SQLite3):", ""]
    for table in db_schema.get("tables", []):
        ddl = table.get("ddl", "").strip()
        if ddl:
            parts.append(ddl)
            parts.append("")

        indexes = table.get("indexes", [])
        for idx in indexes:
            if not isinstance(idx, str):
                logger.warning(f"Invalid index: {idx}, type: {type(idx)}, force to string...")
                idx = str(idx)
            idx_stmt = idx.strip()
            if idx_stmt:
                parts.append(idx_stmt)
        if indexes:
            parts.append("")

    while parts and parts[-1] == "":
        parts.pop()

    return "\n".join(parts)


def tools_robust_json_loads(s: str) -> dict:
    if isinstance(s, Path):
        s = s.as_posix()
        is_path = True
    else:
        is_path = False

    if isinstance(s, str) and os.path.exists(s) and (s.endswith('.json') or s.endswith('.jsonl')) or is_path:
        s = open(s, 'r').read()
    
    if isinstance(s, str) and s.strip() == "":
        return {}

    if isinstance(s, list):
        return [tools_robust_json_loads(item) for item in s]
    
    s = s.encode('utf-8', errors='ignore').decode('utf-8')
    s = s.strip()

    try:
        return orjson.loads(s)
    except Exception:
        pass
    
    try:
        return json.loads(s)
    except Exception:
        pass


    s = _sanitize_for_json_utf8(s)

    decoded = json_repair.loads(s)

    if decoded == "":
        return {}
    
    return decoded

def normalize_scenario_name(scenario: str) -> str:
    s = scenario.lower()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').strip()
    return s


def tools_jsonl_load(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

def tools_jsonl_save(data: list[dict], path: str, append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def tools_token_count(text: str, model: str) -> int:
    try:
        enc = encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        logger.warning(f"Failed to get token count for model {model}, fallback to length of text")
        return len(text)


def json_default(obj):
    return str(obj)

def tools_json_save(data: dict | list, path: str):
    with open(path, 'w', encoding='utf-8', errors='replace') as f:
        try:
            json.dump(data, f, indent=4, ensure_ascii=False, default=json_default)
        except Exception as e:
            print(f"Failed to save JSON to {path}: {e}, fallback to repair_json")
            data = repair_json(str(data), return_objects=True)
            json.dump(data, f, indent=4, ensure_ascii=False, default=json_default)


def normalize_azure_url(base_url: str) -> str:
    """Normalize Azure OpenAI base URLs to include /openai/v1 suffix.
    After normalization, we can directly use the OpenAI client to interact with the Azure OpenAI endpoint.
    """
    if "openai.azure.com" in base_url or "services.ai.azure.com" in base_url:
        stripped = base_url.rstrip("/")
        if not stripped.endswith("/openai/v1"):
            return stripped + "/openai/v1"
    return base_url


def resolve_llm_config(
    api_url_override: str | None = None,
    model_override: str | None = None,
    require_model: bool = False,
) -> tuple[str, str, str]:
    """Resolve LLM API URL, API key, and model from environment variables.

    Supports Azure OpenAI and standard OpenAI providers via env vars:
      - AWM_SYN_LLM_PROVIDER ("azure" | "openai")
      - AZURE_ENDPOINT_URL / AZURE_OPENAI_API_KEY  (for azure)
      - OPENAI_BASE_URL / OPENAI_API_KEY            (for openai)
      - AWM_SYN_OVERRIDE_MODEL                       (model name)

    Args:
        api_url_override: if given, used directly as base_url (still normalized for Azure).
        model_override: if given, used instead of AWM_SYN_OVERRIDE_MODEL env var.
        require_model: if True, raises ValueError when no model can be resolved.

    Returns:
        (api_url, api_key, model)
    """
    provider = os.environ.get("AWM_SYN_LLM_PROVIDER", "").lower()

    # --- resolve base URL ---
    if api_url_override:
        api_url = normalize_azure_url(api_url_override)
    elif provider == "azure":
        azure_endpoint = os.environ.get("AZURE_ENDPOINT_URL", "")
        if not azure_endpoint:
            raise ValueError(
                "AZURE_ENDPOINT_URL not set.\n"
                "Please set: AWM_SYN_LLM_PROVIDER, AZURE_ENDPOINT_URL, AZURE_OPENAI_API_KEY"
            )
        api_url = normalize_azure_url(azure_endpoint)
    elif os.environ.get("OPENAI_BASE_URL"):
        api_url = os.environ["OPENAI_BASE_URL"]
    else:
        raise ValueError("No LLM API URL provided.")

    # --- resolve API key ---
    if provider == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "EMPTY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    # --- resolve model ---
    model = model_override or os.environ.get("AWM_SYN_OVERRIDE_MODEL", "")
    if not model:
        if require_model:
            raise ValueError("AWM_SYN_OVERRIDE_MODEL not set and no model override provided")

    return api_url, api_key, model


def sanitize_for_json(obj):
    """Recursively sanitize an object so it can be safely serialized to JSON."""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)


def find_scenario_entry(
    data: list[dict],
    scenario: str,
    task_idx: int | None = None,
) -> dict | None:
    """Find an entry in a JSONL-loaded list by normalized scenario name and optional task_idx.
    Returns:
        The first matching dict, or None.
    """
    scenario_norm = normalize_scenario_name(scenario)
    for entry in data:
        if normalize_scenario_name(entry.get("scenario", "")) != scenario_norm:
            continue
        if task_idx is not None and entry.get("task_idx") != task_idx:
            continue
        return entry
    return None


if __name__ == "__main__":
    pass