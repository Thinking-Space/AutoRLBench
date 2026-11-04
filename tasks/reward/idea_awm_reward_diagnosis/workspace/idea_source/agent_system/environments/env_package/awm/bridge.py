#!/usr/bin/env python3
"""JSON-lines bridge to Agent World Model's Python 3.12 MCP runtime."""

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from awm.core.agent import MCPToolExecutor, format_tools_for_response, parse_call_tool_arguments
from awm.core.server import Config as ServerConfig, _prepare_database, start_server_process
from awm.core.verify import run_verifier, run_llm_judge
from awm.tools import find_scenario_entry, get_random_available_port, tools_jsonl_load


# Reward contract used by the AWM RL environment.  Unknown verifier/judge
# outcomes intentionally receive zero rather than being treated as failures.
REWARD_BY_TYPE = {
    "complete": 1.0,
    "completed": 1.0,
    "incomplete": 0.1,
    "partially_completed": 0.1,
    "partial": 0.1,
    # Pure-code verifiers use `others` for an unsuccessful completion check;
    # unsuccessful/unknown outcomes are agent/environment errors and receive
    # zero task reward.
    "others": 0.0,
    "agent_error": 0.0,
    "environment_error": 0.0,
    "server_error": 0.0,
    "judge_error": 0.0,
    "format_error": -1.0,
    "tool_not_found": -1.0,
    "invalid_args": -1.0,
}

# Judge labels that count as a successful / partial task outcome.  Terminal
# shaping below only rewrites these; environment/agent errors stay as-is.
_COMPLETE_TYPES = {"complete", "completed"}
_INCOMPLETE_TYPES = {"incomplete", "partially_completed", "partial"}
# Identical successful tool calls (same name + arguments) this many times in a
# row end the episode and run the verifier instead of letting the agent loop.
DEFAULT_REPEAT_SUCCESS_TERMINATE_K = 5


@dataclass
class Session:
    scenario: str
    task_idx: int
    task: str
    output_dir: str
    db_path: str
    server: object | None
    mcp: MCPToolExecutor | None
    tools_text: str
    verifier: dict
    steps: int = 0
    done: bool = False
    trajectory: list[dict] | None = None
    startup_error: str | None = None
    # Consecutive MCP service failures are tracked separately from malformed
    # model actions.  A repeated identical failure indicates an environment
    # problem rather than an agent error.
    last_mcp_error: str | None = None
    mcp_error_streak: int = 0
    last_success_call_key: str | None = None
    success_call_streak: int = 0


class Bridge:
    def __init__(self, config: dict):
        self.root = Path(config["awm_root"])
        self.data = Path(config["data_root"])
        self.max_steps = int(config["max_steps"])
        self.tool_timeout = float(config.get("tool_timeout", 60))
        self.reward_mode = str(config.get("reward_mode", "code")).lower()
        if self.reward_mode not in {"code", "sql"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}; expected code or sql")
        self.judge_base_url = str(config.get("judge_base_url", ""))
        self.judge_api_key = str(config.get("judge_api_key", "EMPTY"))
        self.judge_model = str(config.get("judge_model", ""))
        self.judge_timeout = float(config.get("judge_timeout", 120))
        self.judge_max_retries = int(config.get("judge_max_retries", 3))
        self.judge_concurrency = min(10, max(1, int(config.get("judge_concurrency", 10))))
        self.judge_semaphore = asyncio.Semaphore(self.judge_concurrency)
        # Starting hundreds of MCP subprocesses at once can starve the
        # readiness probes and make otherwise healthy servers miss the window.
        self.startup_concurrency = max(1, int(config.get("startup_concurrency", 32)))
        self.startup_semaphore = asyncio.Semaphore(self.startup_concurrency)
        self.repeat_success_terminate_k = max(
            2, int(config.get("repeat_success_terminate_k", DEFAULT_REPEAT_SUCCESS_TERMINATE_K))
        )
        work_root = config.get("work_root")
        self.work_root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="seed_awm_"))
        self.work_root.mkdir(parents=True, exist_ok=True)
        # Keep the complete hosted LLM-as-judge response independently of the
        # rollout tensors.  The trainer only propagates a subset of env info,
        # so this JSONL is the authoritative per-trajectory judge trace.
        self.judge_log_path = self.work_root / "llm_judge.jsonl"
        # The verifier reads these in the isolated AWM subprocess.
        os.environ["AWM_JUDGE_TIMEOUT"] = str(self.judge_timeout)
        os.environ["AWM_JUDGE_MAX_RETRIES"] = str(self.judge_max_retries)
        self.sessions: list[Session] = []
        verifier_file = "gen_verifier.pure_code.jsonl" if self.reward_mode == "code" else "gen_verifier.jsonl"
        self.verifiers = tools_jsonl_load(str(self.data / verifier_file))

    def _record_judge(
        self,
        session: Session,
        reward_type: str,
        verifier_result: dict,
        judge_result: dict | None,
        shaped_reward_type: str | None = None,
        ended_with_final: bool | None = None,
    ) -> None:
        """Append one complete Judge decision to the bridge-owned JSONL log."""
        entry = {
            "timestamp": time.time(),
            "scenario": session.scenario,
            "task_idx": session.task_idx,
            "task": session.task,
            "steps": len(session.trajectory or []),
            "reward_type": reward_type,
            "verifier_result": verifier_result,
            "llm_judge": judge_result,
        }
        if shaped_reward_type is not None:
            entry["shaped_reward_type"] = shaped_reward_type
        if ended_with_final is not None:
            entry["ended_with_final"] = ended_with_final
        with self.judge_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    async def _stop_server(self, proc):
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
                await asyncio.to_thread(proc.wait, 5)
            except Exception:
                pass
            # awm.core.server launches uvicorn through a shell; clean any group remnants.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pipe in (proc.stdout, proc.stderr):
            if pipe:
                pipe.close()

    async def close(self):
        await self.close_sessions()

    async def close_sessions(self):
        for session in self.sessions:
            proc = session.server
            await self._stop_server(proc)
        self.sessions = []

    async def _start_one(self, index: int, item: dict) -> Session:
        scenario = str(item["scenario"])
        task_idx = int(item["task_idx"])
        last_error = None
        for attempt in range(1, 4):
            output_dir = self.work_root / f"env_{index}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            server = None
            try:
                server_config = ServerConfig(
                    scenario=scenario,
                    envs_load_path=str(self.data / "gen_envs.jsonl"),
                    db_schema_path=str(self.data / "gen_db.jsonl"),
                    sample_path=str(self.data / "gen_sample.jsonl"),
                    temp_server_path=str(output_dir / "server_runtime.py"),
                )
                db_path = _prepare_database(server_config, str(output_dir))
                port = get_random_available_port()
                server = start_server_process(
                    scenario, str(self.data / "gen_envs.jsonl"), db_path, port,
                    output_dir=str(output_dir), temp_server_path=server_config.temp_server_path,
                )
                ready = False
                for _ in range(120):
                    if server.poll() is not None:
                        break
                    try:
                        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                        writer.close()
                        await writer.wait_closed()
                        ready = True
                        break
                    except OSError:
                        await asyncio.sleep(0.5)
                if not ready:
                    raise RuntimeError(f"MCP server failed to start for {scenario} task {task_idx}")
                mcp = MCPToolExecutor(f"http://127.0.0.1:{port}/mcp", timeout=self.tool_timeout)
                tools = []
                for tool_attempt in range(10):
                    tools = await mcp.list_tools()
                    if any(tool.get("name") != "list_tools" for tool in tools):
                        break
                    await asyncio.sleep(min(0.5 * (tool_attempt + 1), 2.0))
                if not any(tool.get("name") != "list_tools" for tool in tools):
                    raise RuntimeError(f"MCP business tools did not register for {scenario} task {task_idx}")
                verifier = find_scenario_entry(self.verifiers, scenario, task_idx=task_idx)
                if not verifier:
                    raise RuntimeError(f"Verifier not found for {scenario} task {task_idx}")
                return Session(
                    scenario=scenario, task_idx=task_idx, task=str(item["task"]),
                    output_dir=str(output_dir), db_path=db_path, server=server, mcp=mcp,
                    tools_text=format_tools_for_response(tools), verifier=verifier, trajectory=[],
                )
            except BaseException as exc:
                last_error = exc
                await self._stop_server(server)
                if attempt < 3:
                    await asyncio.sleep(0.5 * attempt)

        # Keep this item in the vector so the rollout batch remains aligned.
        return Session(
            scenario=scenario, task_idx=task_idx, task=str(item["task"]),
            output_dir=str(self.work_root / f"env_{index}"), db_path="", server=None, mcp=None,
            tools_text="", verifier={}, done=True, trajectory=[],
            startup_error=f"MCP startup failed after 3 attempts: {last_error}",
        )

    async def reset(self, items: list[dict]) -> dict:
        await self.close()

        async def start_limited(index: int, item: dict):
            async with self.startup_semaphore:
                return await self._start_one(index, item)

        results = await asyncio.gather(
            *(start_limited(i, item) for i, item in enumerate(items)),
            return_exceptions=True,
        )
        self.sessions = [result for result in results if isinstance(result, Session)]
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            # Unexpected bridge-level errors are still fatal; individual task
            # startup failures are represented by placeholder sessions above.
            await self.close()
            raise RuntimeError(f"Failed to start {len(errors)} AWM environment(s): {errors[0]}")
        return {"tasks": [session.task for session in self.sessions]}

    async def _verify(self, session: Session, final_answer: str) -> tuple[float, dict, dict, dict | None, str]:
        reward_type, details = run_verifier(
            session.verifier, self.reward_mode,
            str(Path(session.output_dir) / "initial.db"), session.db_path, final_answer,
        )
        judge_result = None
        if self.reward_mode == "sql" and reward_type != "judge_error":
            if not self.judge_base_url or not self.judge_model:
                reward_type = "judge_error"
                judge_result = {"error": "sql reward mode requires judge_base_url and judge_model"}
            else:
                async with self.judge_semaphore:
                    reward_type, judge_result = await run_llm_judge(
                        task=session.task, verifier_result=details,
                        llm_base_url=self.judge_base_url, llm_api_key=self.judge_api_key,
                        llm_model=self.judge_model, trajectory=session.trajectory or [],
                    )
        info = {"reward_type": reward_type, "verify_result": details}
        if judge_result is not None:
            info["llm_judge"] = judge_result
        normalized_type = str(reward_type).strip().lower().replace("-", "_").replace(" ", "_")
        return REWARD_BY_TYPE.get(normalized_type, 0.0), info, details, judge_result, reward_type

    @staticmethod
    def _normalize_reward_type(reward_type: str | None) -> str:
        return str(reward_type or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _success_call_key(name: str, tool_name: str | None, tool_args) -> str | None:
        """Stable identity for a successful list_tools / call_tool action."""
        if name == "list_tools":
            return "list_tools"
        if name != "call_tool" or not tool_name:
            return None
        try:
            args_repr = json.dumps(tool_args, sort_keys=True, default=str, ensure_ascii=False)
        except TypeError:
            args_repr = str(tool_args)
        return f"call_tool:{tool_name}:{args_repr}"

    def _shape_terminal_reward(self, name: str, reward: float, verify_info: dict) -> tuple[float, dict]:
        """Rewrite complete/incomplete when the episode did not end on a final answer.

        complete + last action is still a tool call -> 0.5
        incomplete + no final answer -> -1.0
        Environment / judge / format errors are left unchanged.
        """
        if name == "final":
            return reward, verify_info
        original = self._normalize_reward_type(verify_info.get("reward_type"))
        if original in _COMPLETE_TYPES:
            verify_info = {
                **verify_info,
                "judge_reward_type": original,
                "reward_type": "complete_no_final",
            }
            return 0.5, verify_info
        if original in _INCOMPLETE_TYPES:
            verify_info = {
                **verify_info,
                "judge_reward_type": original,
                "reward_type": "incomplete_no_final",
            }
            return -1.0, verify_info
        return reward, verify_info

    async def _step_one(self, session: Session, action: dict) -> dict:
        if session.startup_error:
            return {
                "observation": f"Environment unavailable: {session.startup_error}",
                "reward": 0.0, "done": True, "action_valid": True,
                "reward_type": "environment_error",
            }
        if session.done:
            return {"observation": "", "reward": 0.0, "done": True}
        session.steps += 1
        name = action.get("name")
        # A malformed tool call is a format failure: stop immediately and
        # assign the contract's -1.0 step reward.  It must not continue until
        # max_steps or enter task-level verification.
        format_failure = name in {"invalid", None} or name not in {"final", "list_tools", "call_tool", "invalid"}
        done = name == "final" or session.steps >= self.max_steps or format_failure
        response = ""
        action_valid = True
        action_error = None
        mcp_error_key = None
        tool_name, tool_args = None, None
        if name == "final":
            # Plain final answers are valid terminal actions.  The content is
            # passed to the verifier below.
            response = str(action.get("content", ""))
        elif name == "list_tools":
            response = session.tools_text
        elif name == "call_tool":
            try:
                tool_name, tool_args = parse_call_tool_arguments(action.get("arguments"))
            except Exception:
                tool_name, tool_args = None, None
            if not tool_name:
                response = "Error: call_tool requires tool_name and JSON arguments"
                action_valid = False
                action_error = "invalid_args"
            else:
                try:
                    response = await session.mcp.call_tool(tool_name, tool_args)
                except asyncio.TimeoutError:
                    response = f"Error: MCP tool call timed out after {self.tool_timeout}s"
                    action_valid = False
                    mcp_error_key = response
                except Exception as exc:
                    response = f"Error: {type(exc).__name__}: {exc}"
                    action_valid = False
                    mcp_error_key = response
                else:
                    # Some MCP implementations return an error payload as a
                    # string instead of raising.  Treat those responses as
                    # service errors too, while leaving ordinary tool output
                    # untouched.
                    if isinstance(response, str) and response.startswith("Error:"):
                        action_valid = False
                        mcp_error_key = response
        elif name == "invalid":
            response = "Error: output one valid <tool_call> block, or a plain final answer."
            action_valid = False
            action_error = "format_error"
        else:
            action_valid = False
            action_error = "format_error"

        if mcp_error_key is not None:
            if session.last_mcp_error == mcp_error_key:
                session.mcp_error_streak += 1
            else:
                session.last_mcp_error = mcp_error_key
                session.mcp_error_streak = 1
            # Two consecutive identical MCP failures are treated as an
            # environment failure.  Do not attribute this to the agent.
            if session.mcp_error_streak >= 2:
                action_error = "environment_error"
                done = True
            session.last_success_call_key = None
            session.success_call_streak = 0
        else:
            session.last_mcp_error = None
            session.mcp_error_streak = 0

        # Malformed arguments and invalid output remain terminal agent errors.
        if action_error is not None and action_error != "environment_error":
            done = True
            session.last_success_call_key = None
            session.success_call_streak = 0

        # Identical successful tool calls in a row are treated as looping:
        # stop and send the trajectory to the verifier instead of burning
        # the remaining budget on the same call.  MCP errors are not
        # successful calls and must not count toward this streak.
        if action_error is None and mcp_error_key is None and name in {"list_tools", "call_tool"}:
            call_key = self._success_call_key(name, tool_name, tool_args)
            if call_key and call_key == session.last_success_call_key:
                session.success_call_streak += 1
            else:
                session.last_success_call_key = call_key
                session.success_call_streak = 1
            if session.success_call_streak >= self.repeat_success_terminate_k:
                done = True
        elif action_error is None and mcp_error_key is None:
            session.last_success_call_key = None
            session.success_call_streak = 0

        session.trajectory.append({
            "iteration": session.steps, "role": "assistant",
            "content": str(action.get("content", "")),
            "tool_calls": [action] if name in {"list_tools", "call_tool", "invalid"} else [],
            "tool_response": {"content": response},
        })

        # Only normally completed trajectories are sent to the task verifier.
        # Format failures are terminal immediately and bypass task judging.
        reward = 0.0
        verify_info = {}
        if action_error is not None:
            reward = REWARD_BY_TYPE[action_error]
            verify_info["reward_type"] = action_error
        elif done:
            verified_reward, verify_info, details, judge_result, raw_reward_type = await self._verify(
                session, str(action.get("content", "")),
            )
            reward, verify_info = self._shape_terminal_reward(name, verified_reward, verify_info)
            if self.reward_mode == "sql":
                self._record_judge(
                    session, raw_reward_type, details, judge_result,
                    shaped_reward_type=str(verify_info.get("reward_type", raw_reward_type)),
                    ended_with_final=name == "final",
                )
            session.done = True
        if action_error is not None:
            session.done = True
        return {
            "observation": response,
            "reward": reward,
            "done": done,
            "action_valid": action_valid,
            **verify_info,
        }

    async def step(self, actions: list[dict]) -> dict:
        if len(actions) != len(self.sessions):
            raise ValueError(f"Expected {len(self.sessions)} actions, got {len(actions)}")
        results = await asyncio.gather(*(self._step_one(s, a) for s, a in zip(self.sessions, actions)))
        return {"results": results}


async def main() -> None:
    config = json.loads(sys.stdin.readline())
    bridge = Bridge(config)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request.pop("command")
                if command == "reset":
                    result = await bridge.reset(request["items"])
                elif command == "step":
                    result = await bridge.step(request["actions"])
                elif command == "close_sessions":
                    await bridge.close_sessions()
                    result = {"closed": True}
                elif command == "close":
                    await bridge.close()
                    result = {"closed": True}
                else:
                    raise ValueError(f"Unknown command: {command}")
                print(json.dumps({"ok": True, **result}, ensure_ascii=False), flush=True)
                if command == "close":
                    break
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        await bridge.close()


if __name__ == "__main__":
    asyncio.run(main())
