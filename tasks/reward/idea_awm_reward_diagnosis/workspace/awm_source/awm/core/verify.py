"""
Verify agent run outputs.

Supports two verification modes:
- "sql": Run verifier code to extract DB state info, then use LLM to judge. This mode is recommended for most cases.
- "code": Run verifier code that deterministically returns "complete" or "others".
"""

import inspect
import json
import os
import re
import sqlite3
import asyncio
import httpx
from dataclasses import dataclass
from textwrap import dedent
from enum import Enum

from loguru import logger
from openai import AsyncOpenAI

from awm.tools import (
    tools_jsonl_load,
    tools_json_save,
    normalize_azure_url,
    resolve_llm_config,
    sanitize_for_json,
    find_scenario_entry,
)


class VerificationMode(Enum):
    sql = "sql"  # code-augmented verification (returns info dict for LLM judgment)
    code = "code" # purely code-based verification (returns complete/others)

@dataclass
class Config:
    # Path to the agent run output directory (contains trajectory.json, initial.db, final.db)
    input: str
    # optional, path to the initial database path, if specified, will override the input
    init_db_path: str | None = None
    # optional, path to the final database path, if specified, will override the input
    final_db_path: str | None = None
    # Verification mode: "sql" or "code", sql mode is the recommended code-augmented LLM-as-a-Judge mode.
    mode: VerificationMode = VerificationMode.sql
    # Path to verifier data file (sql mode). Defaults to outputs/gen_verifier.jsonl
    verifier_path: str | None = None
    # Path to verifier data file (code mode). Defaults to outputs/gen_verifier.pure_code.jsonl
    verifier_code_path: str | None = None

    def pre_process(self):
        if self.mode == VerificationMode.sql:
            # check llm environment variables
            provider = os.environ.get("AWM_SYN_LLM_PROVIDER", "").lower()
            if provider == "azure":
                missing = []
                if not os.environ.get("AZURE_ENDPOINT_URL"):
                    missing.append("AZURE_ENDPOINT_URL")
                if not os.environ.get("AZURE_OPENAI_API_KEY"):
                    missing.append("AZURE_OPENAI_API_KEY")
                if missing:
                    raise ValueError(
                        f"SQL verification mode requires Azure LLM config. Missing env vars: {', '.join(missing)}.\n"
                        "Please set: AWM_SYN_LLM_PROVIDER=azure, AZURE_ENDPOINT_URL, AZURE_OPENAI_API_KEY"
                    )
            else:
                if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
                    raise ValueError(
                        "SQL verification mode requires LLM config. Missing env vars.\n"
                        "Please set either:\n"
                        "  - AWM_SYN_LLM_PROVIDER=azure + AZURE_ENDPOINT_URL + AZURE_OPENAI_API_KEY\n"
                        "  - OPENAI_API_KEY (+ optional OPENAI_BASE_URL)"
                    )
            if not os.environ.get("AWM_SYN_OVERRIDE_MODEL"):
                raise ValueError(
                    "SQL verification mode requires AWM_SYN_OVERRIDE_MODEL to be set.\n"
                    "Example: export AWM_SYN_OVERRIDE_MODEL=gpt-5"
                )



def _call_verifier_func(verify_func, initial_db_path, final_db_path, final_answer=None):
    """Call a verifier function with signature-aware argument passing."""
    sig = inspect.signature(verify_func)
    params = sig.parameters

    kwargs = {}
    if "initial_db_path" in params:
        kwargs["initial_db_path"] = initial_db_path
    if "final_db_path" in params:
        kwargs["final_db_path"] = final_db_path
    if "final_answer" in params:
        kwargs["final_answer"] = final_answer or ""

    if not kwargs and len(params) >= 2:
        args = [initial_db_path, final_db_path]
        if len(params) >= 3:
            args.append(final_answer or "")
        return verify_func(*args)

    return verify_func(**kwargs)


def execute_sql_verifier(verifier_code, function_name, initial_db_path, final_db_path):
    """Execute SQL-mode verifier code that compares initial vs final DB state."""
    if not os.path.exists(initial_db_path):
        return {"execution_status": "error", "error_message": f"Initial DB not found: {initial_db_path}"}
    if not os.path.exists(final_db_path):
        return {"execution_status": "error", "error_message": f"Final DB not found: {final_db_path}"}

    original_modes = {}
    for path in [initial_db_path, final_db_path]:
        try:
            original_modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except Exception:
            pass

    try:
        namespace = {
            "sqlite3": sqlite3,
            "json": json,
            "os": os,
            "__builtins__": __builtins__,
        }
        exec(verifier_code, namespace)

        verify_func = namespace.get(function_name)
        if not verify_func:
            return {"execution_status": "error", "error_message": f"Function '{function_name}' not found"}

        result = _call_verifier_func(verify_func, initial_db_path, final_db_path)

        try:
            json.dumps(result)
        except TypeError:
            result = sanitize_for_json(result)

        return result

    except Exception as e:
        return {"execution_status": "error", "error_message": f"Execution error: {e}"}
    finally:
        for path, mode in original_modes.items():
            try:
                os.chmod(path, mode)
            except Exception:
                pass


def execute_code_verifier(verifier_code, function_name, initial_db_path, final_db_path, final_answer=None):
    """Execute code-mode verifier that returns {"result": "complete"|"others"}."""
    if not os.path.exists(initial_db_path):
        return {"result": "others", "execution_status": "error", "error_message": f"Initial DB not found: {initial_db_path}"}
    if not os.path.exists(final_db_path):
        return {"result": "others", "execution_status": "error", "error_message": f"Final DB not found: {final_db_path}"}

    original_modes = {}
    for path in [initial_db_path, final_db_path]:
        try:
            original_modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except Exception:
            pass

    try:
        namespace = {
            "sqlite3": sqlite3,
            "json": json,
            "os": os,
            "re": re,
            "__builtins__": __builtins__,
        }
        exec(verifier_code, namespace)

        verify_func = namespace.get(function_name)
        if not verify_func:
            return {"result": "others", "execution_status": "error", "error_message": f"Function '{function_name}' not found"}

        result = _call_verifier_func(verify_func, initial_db_path, final_db_path, final_answer)

        if not isinstance(result, dict) or "result" not in result:
            return {"result": "others", "execution_status": "error", "error_message": f"Invalid return format: {type(result).__name__}"}

        result_value = result.get("result", "others")
        if result_value not in ("complete", "others"):
            result_value = "others"

        return {"result": result_value, "execution_status": "success", "raw_result": result}

    except Exception as e:
        return {"result": "others", "execution_status": "error", "error_message": f"Execution error: {e}"}
    finally:
        for path, mode in original_modes.items():
            try:
                os.chmod(path, mode)
            except Exception:
                pass


def run_verifier(verifier_entry, verifier_mode, initial_db_path, final_db_path, final_answer=None):
    """Run the appropriate verifier and return (reward_type, details)."""
    verification = verifier_entry.get("verification", {})
    code = verification.get("code", "")

    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return "judge_error", {"error": "No valid verifier code found"}

    default_func = "verify_task_completion" if verifier_mode == "code" else "verify_task"
    func_name = default_func
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            func_name = line.split("(")[0].replace("def ", "").strip()
            break

    if verifier_mode == "code":
        result = execute_code_verifier(code, func_name, initial_db_path, final_db_path, final_answer)
        if result.get("execution_status") == "error":
            return "judge_error", result
        return result.get("result", "others"), result

    result = execute_sql_verifier(code, func_name, initial_db_path, final_db_path)
    if isinstance(result, dict) and result.get("execution_status") == "error":
        return "judge_error", result

    return "incomplete", result


async def run_llm_judge(task, verifier_result, llm_base_url, llm_api_key, llm_model, trajectory=None):
    """Run LLM-as-a-judge combining agent trajectory AND SQL verification results."""
    if not llm_base_url or not llm_model:
        return "judge_error", {"error": "LLM endpoint not configured for sql verifier mode"}

    llm_base_url = normalize_azure_url(llm_base_url)

    try:
        # Hosted judges can take longer than the SDK's connect/read defaults,
        # and inherited proxy settings are often unreliable on training nodes.
        timeout = float(os.environ.get("AWM_JUDGE_TIMEOUT", "120"))
        max_retries = max(0, int(os.environ.get("AWM_JUDGE_MAX_RETRIES", "3")))
        http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        client = AsyncOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

        system_prompt = dedent("""\
            You are an impartial evaluator of automated agent task results with access to database verification. Based on the provided JSON trajectory AND the Python verification results from querying the database, decide the task outcome. This trajectory is generated by an MCP agent on a simulated simplified environment. The environment provides a set of MCP tools to help the agent complete the task.

            Input:
                task_json: dict containing task fields such as user task and agent execution_history.
                verification_json: dict containing Python code, reasoning, success_criteria, failure_criteria, and execution results that verified the database state from the Python function

            Output:
                You only output UTF-8 encoded string, avoid any emoji or special characters. You only output English text.

            Classification categories:
                - complete: all required steps and closure actions were successfully executed, AND the database state confirms the task was completed
                - incomplete: partial progress or the database state shows the task is not fully completed
                - server_error: the agent is blocked by MCP server/environment error, e.g., 5xx errors such as "Internal Server Error". Or the MCP server cannot process the valid tool call and return valid results. This can block the agent from completing the task.
                - agent_error: the agent made mistakes, invalid parameters, or missing required data without recovery, failed to complete the user's instruction.

            Priority order for classification:
                1) complete (trajectory shows success AND database confirms it)
                2) server_error (due to the MCP server/environment error)
                3) agent_error (agent-side issue, e.g., invalid mcp_tool_call arguments, hallucination, agent mistakes)
                4) incomplete (everything else unfinished or database state doesn't match expected outcome)

            Key considerations:
            - The verification_json contains checks performed on the database states. You can read the verification code to understand what checks were performed on the database states.
            - The verification_json contains the execution results of the verification code. You can use the execution results to help you judge the task completion.
            - The verification results may be empty or error, or even the verification code itself is inaccurate. You should not fully rely on the verification results. You need to comprehensively consider the trajectory information to help you judge the task completion.

            Output format (must be valid JSON, no markdown fences, no additional commentary):
                {
                  "reasoning": "<concise explanation considering both trajectory and verification code execution results>",
                  "confidence_score": [0-100, 0-100, 0-100, 0-100] for complete, incomplete, server_error, agent_error respectively,
                  "classification": "<one_of_[complete, incomplete, server_error, agent_error]>",
                  "evidence": {
                    "status": "<original result.status>",
                    "iterations": <int>,
                    "error_signals": ["<important error messages or codes>"],
                    "last_actions": ["<summaries of last few actions>"],
                    "database_verification": "<summary of what the database state changed based on code execution results of verification>"
                  }
                }""")

        task_payload = {
            "user_task": task,
            "actual_execution_steps": len(trajectory) if trajectory else 0,
            "trajectory": trajectory or [],
        }

        verification_json = sanitize_for_json({
            "code_execution_result": verifier_result,
        })
        try:
            verification_json_str = json.dumps(verification_json, ensure_ascii=False, indent=2, default=str)
        except Exception:
            verification_json_str = str(verification_json)

        user_prompt = (
            f"task_json:\n"
            f"{json.dumps(task_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
            f"verification_json:\n"
            f"{verification_json_str}"
        )

        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_completion_tokens=2048,
            )
        finally:
            await http_client.aclose()

        content = response.choices[0].message.content or ""
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    return "judge_error", {"error": f"Failed to parse LLM response: {content}"}
            else:
                return "judge_error", {"error": f"Failed to parse LLM response: {content}"}

        classification = result.get("classification", "judge_error").lower().strip()
        valid = {"complete", "incomplete", "server_error", "agent_error", "judge_error"}
        if classification not in valid:
            classification = "judge_error"

        return classification, result

    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return "judge_error", {"error": str(e)}




async def run_verify(config: Config):
    """Run verification on an agent run directory."""
    run_dir = config.input
    mode = config.mode

    if isinstance(mode, VerificationMode):
        mode = mode.value
    if mode not in ("sql", "code"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'sql' or 'code'.")

    # Load trajectory to get scenario and task info
    trajectory_path = os.path.join(run_dir, "trajectory.json")
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"trajectory.json not found in {run_dir}")

    with open(trajectory_path) as f:
        trajectory_data = json.load(f)

    scenario = trajectory_data.get("scenario")
    task_id = trajectory_data.get("task_id")
    task = trajectory_data.get("task", "")
    trajectory = trajectory_data.get("trajectory", [])
    messages = trajectory_data.get("messages", [])

    if scenario is None or task_id is None:
        raise ValueError("trajectory.json must contain 'scenario' and 'task_id' fields")

    # Check database files (CLI overrides take priority over run_dir defaults)
    initial_db_path = config.init_db_path or os.path.join(run_dir, "initial.db")
    final_db_path = config.final_db_path or os.path.join(run_dir, "final.db")

    if not os.path.exists(initial_db_path):
        raise FileNotFoundError(f"Initial database not found: {initial_db_path}")
    if not os.path.exists(final_db_path):
        raise FileNotFoundError(f"Final database not found: {final_db_path}")

    # Load verifier data
    if mode == "code":
        verifier_file = config.verifier_code_path or "./outputs/gen_verifier.pure_code.jsonl"
    else:
        verifier_file = config.verifier_path or "./outputs/gen_verifier.jsonl"

    if not os.path.exists(verifier_file):
        raise FileNotFoundError(f"Verifier file not found: {verifier_file}")

    verifier_data = tools_jsonl_load(verifier_file)
    verifier_entry = find_scenario_entry(verifier_data, scenario, task_idx=task_id)

    if not verifier_entry:
        raise ValueError(f"No verifier found for scenario={scenario}, task_id={task_id} in {verifier_file}")

    logger.info(f"Running {mode} verification for scenario={scenario}, task_id={task_id}")

    # Extract final answer from trajectory (last assistant message)
    final_answer = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
            break

    # Run verifier
    reward_type, verify_result = run_verifier(
        verifier_entry, mode, initial_db_path, final_db_path, final_answer
    )

    logger.info(f"Verifier result: reward_type={reward_type}")
    logger.info(f"Verify result: {json.dumps(verify_result, ensure_ascii=False, default=str)[:500]}")

    output = {
        "scenario": scenario,
        "task_id": task_id,
        "task": task,
        "mode": mode,
        "reward_type": reward_type,
        "verify_result": verify_result,
    }

    # For sql mode, also run LLM judge
    if mode == "sql" and reward_type != "judge_error":
        logger.info("Running LLM judge for sql mode...")
        llm_base_url, llm_api_key, llm_model = resolve_llm_config(require_model=True)

        classification, judge_result = await run_llm_judge(
            task=task,
            verifier_result=verify_result,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            trajectory=trajectory,
        )
        output["llm_judge"] = {
            "classification": classification,
            "judge_result": judge_result,
        }
        logger.info(f"LLM judge classification: {classification}")

    # Save verification result
    output_path = os.path.join(run_dir, f"verify.{mode}.json")
    tools_json_save(output, output_path)
    logger.info(f"Saved verification result to {output_path}")

    return output


def run(config: Config):
    asyncio.run(run_verify(config))


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)
