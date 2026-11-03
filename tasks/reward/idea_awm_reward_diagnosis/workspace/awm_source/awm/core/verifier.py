import json
import sqlite3
import os
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

from awm.gpt import GPTClient
from awm.tools import tools_robust_json_loads, normalize_scenario_name, tools_jsonl_save, tools_jsonl_load, dump_sqlite_to_string
from awm.prompts import (
    VERIFIER_SQL_GENERATION_SYSTEM_PROMPT,
    VERIFIER_SQL_GENERATION_USER_PROMPT,
    CODE_VERIFICATION_SYSTEM_PROMPT,
    CODE_VERIFICATION_USER_PROMPT,
)


class VerificationMode(Enum):
    sql = "sql"  # code-augmented verification (returns info dict for LLM judgment)
    code = "code" # purely code-based verification (returns complete/others)


@dataclass
class Config:
    input_task: str
    output: str
    database_dir: str = './outputs/databases'
    mode: VerificationMode = VerificationMode.sql
    model: str = "your-llm-model-name"
    limit: int | None = None # limit the number of tasks to process
    allowed_scenarios: list[str] | None = None  # Comma-separated list of allowed scenarios
    max_retry: int = 4
    batch_size: int = 128
    max_concurrency: int = 32

    def pre_process(self):
        assert os.path.exists(self.input_task), f"Input task file {self.input_task} does not exist"
        assert os.path.exists(self.database_dir), f"Databases folder {self.database_dir} does not exist"
        assert self.output is not None and self.output.endswith('.jsonl'), "Output path .jsonl is required"

        if self.allowed_scenarios:
            self.allowed_scenarios = {normalize_scenario_name(name) for name in self.allowed_scenarios}
        
        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def _load_db_dump_worker(args: tuple) -> tuple[str, str, str | None]:
    scenario, database_dir = args
    db_path = f"{database_dir}/{normalize_scenario_name(scenario)}.db"
    if not os.path.exists(db_path):
        raise RuntimeError(f"database not found: {db_path}")
    db_dump = dump_sqlite_to_string(db_path)
    return scenario, db_path, db_dump, None


def sql_generation_prompt(scenario: str, task: str, db_dump: str) -> list[dict]:
    return [
        {"role": "system", "content": VERIFIER_SQL_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": VERIFIER_SQL_GENERATION_USER_PROMPT.format(
            scenario_name=scenario,
            user_task=task,
            db_dump=db_dump,
        )},
    ]


def code_verification_prompt(scenario: str, task: str, db_dump: str) -> list[dict]:
    return [
        {"role": "system", "content": CODE_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": CODE_VERIFICATION_USER_PROMPT.format(
            scenario=scenario,
            task=task,
            db_dump=db_dump,
        )},
    ]


def execute_verification_code(
    python_code: str,
    function_name: str,
    initial_db_path: str,
    mode: VerificationMode = VerificationMode.sql,
) -> dict:

    if not os.path.exists(initial_db_path):
        raise RuntimeError(f"Database not found: {initial_db_path}")

    final_db_path = initial_db_path  # use same db for testing
    original_mode = os.stat(initial_db_path).st_mode

    try:
        os.chmod(initial_db_path, 0o444)

        namespace = {
            'sqlite3': sqlite3,
            'json': json,
        }

        exec(python_code, namespace)

        verify_func = namespace.get(function_name)
        if not verify_func:
            return {
                "execution_status": "error",
                "error_message": f"Function '{function_name}' not found in generated code",
            }

        if mode == VerificationMode.sql:
            result = verify_func(initial_db_path=initial_db_path, final_db_path=final_db_path)
        else:
            result = verify_func(
                initial_db_path=initial_db_path,
                final_db_path=final_db_path,
                final_answer="EMPTY RESPONSE FOR TESTING",
            )

        if mode == VerificationMode.code:
            if not isinstance(result, dict) or "result" not in result:
                return {
                    "execution_status": "error",
                    "error_message": f"Invalid return format. Expected dict with 'result' key, got: {type(result).__name__}",
                }
            if result["result"] and isinstance(result["result"], str) and result["result"].lower() not in {"complete", "others"}:
                return {
                    "execution_status": "error",
                    "error_message": f"Invalid result value. Expected 'complete' or 'others', got: {result.get('result')}",
                }

        return {"execution_status": "success", "result": result}

    except Exception as e:
        return {"execution_status": "error", "error_message": f"Execution error: {str(e)}"}
    finally:
        try:
            os.chmod(initial_db_path, original_mode)
        except Exception:
            pass


def load_existing_results(output_path: str) -> dict[str, dict]:
    # for resume
    if not os.path.exists(output_path):
        return {}

    existing_map: dict[str, dict] = {}
    data = tools_jsonl_load(output_path)
    for item in data:
        scenario = normalize_scenario_name(item['scenario'])
        task = item['task']
        verification = verification = item.get('verification', {})
        if verification and verification.get('code') and isinstance(verification['code'], str) and len(verification['code']) > 10:
            existing_map[f"{scenario}::{task}"] = item

    return existing_map


class VerificationCodeGenerator:
    def __init__(self, args: Config):
        self.args = args
        self.gpt_client = GPTClient(
            timeout=1200,
            max_retry_num=3,
            concurrency_limit=self.args.max_concurrency,
        )
        self.pending_results: list[dict] = []

    def _save_pending_results(self):
        if not self.pending_results:
            return
        tools_jsonl_save(self.pending_results, self.args.output, append=True)
        logger.info(f"Saved {len(self.pending_results)} results to {self.args.output}")
        self.pending_results = []

    def _build_generation_request(
        self,
        scenario: str,
        task: str,
        db_dump: str,
        error_history: list[str] | None = None,
    ) -> dict:
        if self.args.mode == VerificationMode.sql:
            messages = sql_generation_prompt(scenario, task, db_dump)
        else:
            messages = code_verification_prompt(scenario, task, db_dump)

        # error history if retrying
        if error_history:
            error_context = "\n\n".join([f"Previous Error #{i+1}:\n{err}" for i, err in enumerate(error_history)])
            messages = [m.copy() for m in messages]
            messages[-1]["content"] += f"\n\n## CRITICAL: Avoid these errors from previous attempts:\n{error_context}"

        return {
            "messages": messages,
            "model": self.args.model,
            "temperature": 1.0,
            "max_tokens": 32000,
        }

    def _build_error_summary_request(self, error_message: str, python_code: str) -> dict:
        """Build a request for error summarization."""
        system_prompt = "You are a Python debugging expert. Analyze the error and provide a concise summary with fix guidance. Keep response under 500 tokens."
        user_prompt = f"""Error Message:
{error_message}

Generated Code:
```python
{python_code}
```

Provide:
1. Root cause of the error
2. Specific fix guidance"""

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": self.args.model,
            "temperature": 1.0,
            "max_tokens": 1024,
        }

    def _finalize_task(self, scenario: str, task: str, task_idx: int, state: dict) -> None:
        generation_result = state["generation_result"]

        result = {
            "scenario": scenario,
            "task_idx": task_idx,
            "task": task,
            "verification": {
                "code": generation_result.get("python_code") if generation_result else None,
                "raw_response": state["raw_response"],
            },
        }
        self.pending_results.append(result)

    def process_tasks(self, task_items: list[dict], existing_results: dict[str, dict]) -> None:
        # collect unique scenarios for DB dump loading
        scenarios = set()
        for item in task_items:
            scenarios.add(normalize_scenario_name(item["scenario"]))

        logger.info(f"Loading database dumps for {len(scenarios)} scenarios...")
        worker_args = [(s, self.args.database_dir) for s in scenarios]
        db_info: dict[str, dict] = {}

        num_workers = min(os.cpu_count() or 4, len(worker_args), 16)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_load_db_dump_worker, arg): arg[0] for arg in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading DB dumps"):
                scenario, db_path, db_dump, error = future.result()
                db_info[scenario] = {"db_path": db_path, "db_dump": db_dump, "error": error}

        # validate existing results by actually executing the code
        default_func_name = "verify_task_completion" if self.args.mode == VerificationMode.code else "verify_task"
        validated_keys: set[str] = set()
        to_validate: list[tuple[str, str, str, str]] = []  # (resume_key, code, db_path, func_name)
        seen_keys: set[str] = set()
        for item in task_items:
            scenario = normalize_scenario_name(item["scenario"])
            for task in item["tasks"]:
                resume_key = f"{scenario}::{task}"
                if resume_key in existing_results and resume_key not in seen_keys:
                    seen_keys.add(resume_key)
                    existing = existing_results[resume_key]
                    code = existing.get('verification', {}).get('code', '')
                    info = db_info.get(scenario)
                    if code and info and info.get("db_path"):
                        # try to extract function name from existing code
                        func_name = default_func_name
                        for line in code.split('\n'):
                            line = line.strip()
                            if line.startswith('def verify_') and '(' in line:
                                func_name = line.split('(')[0].replace('def ', '').strip()
                                break
                        to_validate.append((resume_key, code, info["db_path"], func_name))

        if to_validate:
            logger.info(f"Validating {len(to_validate)} existing results by executing code...")
            for resume_key, code, db_path, func_name in tqdm(to_validate, desc="Validating existing results"):
                exec_result = execute_verification_code(
                    python_code=code,
                    function_name=func_name,
                    initial_db_path=db_path,
                    mode=self.args.mode,
                )
                if exec_result.get("execution_status") == "success":
                    validated_keys.add(resume_key)
            logger.info(f"Validation passed: {len(validated_keys)}/{len(to_validate)}")

        # build task_map: (scenario, task_text) -> list[task_idx]
        task_map: dict[tuple[str, str], list[int]] = {}
        for item in task_items:
            scenario = normalize_scenario_name(item["scenario"])
            for task_idx, task in enumerate(item["tasks"]):
                resume_key = f"{scenario}::{task}"
                if resume_key in validated_keys:
                    continue
                map_key = (scenario, task)
                if map_key not in task_map:
                    task_map[map_key] = []
                task_map[map_key].append(task_idx)

        if not task_map:
            logger.info("all tasks already processed")
            return

        logger.info(f"processing {len(task_map)} unique (scenario, task) pairs")

        # Initialize task states
        max_attempts = self.args.max_retry + 1
        task_states: dict[tuple[str, str], dict] = {}
        for key in task_map:
            task_states[key] = {
                "attempt_count": 0,
                "success": False,
                "generation_result": None,
                "raw_response": None,
                "error_history": [],
                "last_code": None,
                "needs_error_summary": False,
                "pending_error_msg": None,
            }

        pending_keys = list(task_states.keys())
        batch_size = self.args.batch_size

        logger.info(f"processing loop: {len(pending_keys)} tasks, batch_size={batch_size}, max_attempts={max_attempts}")

        while pending_keys:
            batch_keys = pending_keys[:batch_size]
            pending_keys = pending_keys[batch_size:]

            # Separate tasks needing error summary vs ready for generation
            need_error_summary = []
            ready_for_generation = []

            for key in batch_keys:
                state = task_states[key]
                if state["needs_error_summary"] and state["pending_error_msg"]:
                    need_error_summary.append(key)
                else:
                    ready_for_generation.append(key)

            if need_error_summary:
                error_summary_requests = []
                for key in need_error_summary:
                    state = task_states[key]
                    error_summary_requests.append(
                        self._build_error_summary_request(state["pending_error_msg"], state["last_code"])
                    )

                logger.info(f"Summarizing {len(error_summary_requests)} errors...")
                error_summaries = self.gpt_client.batch_chat_completion(error_summary_requests, progress_bar=True)

                for key, summary in zip(need_error_summary, error_summaries):
                    state = task_states[key]
                    state["error_history"].append(summary if summary else "Error summarization failed")
                    state["needs_error_summary"] = False
                    state["pending_error_msg"] = None
                    ready_for_generation.append(key)

            if not ready_for_generation:
                continue

            # generation
            generation_requests = []
            request_keys = []
            for key in ready_for_generation:
                scenario, task = key
                state = task_states[key]
                info = db_info[scenario]

                request = self._build_generation_request(
                    scenario=scenario,
                    task=task,
                    db_dump=info["db_dump"],
                    error_history=state["error_history"] if state["error_history"] else None,
                )
                generation_requests.append(request)
                request_keys.append(key)

            attempt_info = {key: task_states[key]["attempt_count"] + 1 for key in request_keys}
            logger.info(f"generating verification code for {len(generation_requests)} tasks (attempts: {list(attempt_info.values())})")
            responses = self.gpt_client.batch_chat_completion(generation_requests, progress_bar=True)

            # process responses and test code execution
            for key, response in zip(request_keys, responses):
                state = task_states[key]
                scenario, task = key
                info = db_info[scenario]

                state["attempt_count"] += 1
                state["raw_response"] = response
                current_attempt = state["attempt_count"]

                # parse response
                try:
                    result = tools_robust_json_loads(response) if response else {}
                    if not result or "python_code" not in result:
                        state["error_history"].append("Generation failed: Invalid response format")
                        if current_attempt < max_attempts:
                            pending_keys.append(key)
                        else:
                            for task_idx in task_map[key]:
                                self._finalize_task(scenario, task, task_idx, state)
                        continue
                    state["generation_result"] = result
                    state["last_code"] = result.get("python_code", "")
                except Exception as e:
                    state["error_history"].append(f"Failed to parse response: {e}")
                    if current_attempt < max_attempts:
                        pending_keys.append(key)
                    else:
                        for task_idx in task_map[key]:
                            self._finalize_task(scenario, task, task_idx, state)
                    continue

                # test the generated code
                python_code = result.get("python_code", "")
                default_func_name = "verify_task_completion" if self.args.mode == VerificationMode.code else "verify_task"
                function_name = result.get("function_name", default_func_name)

                exec_result = execute_verification_code(
                    python_code=python_code,
                    function_name=function_name,
                    initial_db_path=info["db_path"],
                    mode=self.args.mode,
                )

                if exec_result.get("execution_status") == "success":
                    state["success"] = True
                    logger.debug(f"[{scenario}] Verification code test passed (attempt {current_attempt})")
                    for task_idx in task_map[key]:
                        self._finalize_task(scenario, task, task_idx, state)
                else:
                    error_msg = exec_result.get("error_message", "Unknown execution error")
                    logger.debug(f"[{scenario}] Execution error (attempt {current_attempt}): {error_msg}")

                    if current_attempt < max_attempts:
                        state["needs_error_summary"] = True
                        state["pending_error_msg"] = error_msg
                        pending_keys.append(key)
                    else:
                        state["error_history"].append(error_msg)
                        for task_idx in task_map[key]:
                            self._finalize_task(scenario, task, task_idx, state)

            # Save after each batch
            self._save_pending_results()

            completed = len(task_states) - len(pending_keys)
            logger.info(f"Progress: {completed}/{len(task_states)} tasks completed, {len(pending_keys)} pending")


def run(config: Config):
    logger.info(f"config for generating verification code: {config}")

    existing_results = load_existing_results(config.output)
    task_items = tools_jsonl_load(config.input_task)

    if config.allowed_scenarios:
        task_items = [item for item in task_items if normalize_scenario_name(item["scenario"]) in config.allowed_scenarios]

    if config.limit:
        tasks_per_scenario = config.limit // len(task_items)
        tasks_per_scenario = max(tasks_per_scenario, 1)
        temp = []
        for item in task_items:
            item['tasks'] = item['tasks'][:tasks_per_scenario]
            temp.append(item)
            if sum(len(item['tasks']) for item in temp) >= config.limit:
                break
        task_items = temp

    total_tasks = sum(len(item["tasks"]) for item in task_items)
    already_done = [f"{normalize_scenario_name(task_item['scenario'])}::{task}" in existing_results for task_item in task_items for task in task_item['tasks']]
    already_done = sum(already_done)

    logger.info(f"Total tasks: {total_tasks}, Already done: {already_done}, Full existing results: {len(existing_results)}")

    if total_tasks != 0:
        generator = VerificationCodeGenerator(config)
        generator.process_tasks(task_items, existing_results)

    logger.info(f"Done verification generation!config={config}")

