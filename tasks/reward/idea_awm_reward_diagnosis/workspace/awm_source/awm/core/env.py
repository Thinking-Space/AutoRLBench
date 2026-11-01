from awm.gpt import GPTClient
from awm.prompts import ENVIRONMENT_GENERATION_SYSTEM_PROMPT, ENVIRONMENT_GENERATION_USER_PROMPT
from loguru import logger
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
import subprocess
import shutil
import time
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from awm.tools import tools_robust_json_loads, normalize_scenario_name, format_db_schema, tools_token_count, wait_port_free, wait_for_server, get_random_available_port, tools_jsonl_save, tools_jsonl_load

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"



@dataclass
class Config:
    input_spec: str
    input_db: str
    output: str
    database_dir: str = './outputs/databases'
    model: str = "your-llm-model-name"
    allowed_scenarios: list[str] | None = None  # Comma-separated list of allowed scenarios
    max_retry: int = 4  # Maximum number of retry attempts

    def pre_process(self):
        assert self.output is not None and self.output.endswith('.jsonl'), "Output path .jsonl is required"
        if self.allowed_scenarios:
            self.allowed_scenarios = {normalize_scenario_name(name) for name in self.allowed_scenarios}
        assert self.input_spec and os.path.exists(self.input_spec), f"API spec file {self.input_spec} not found"
        assert self.input_db and os.path.exists(self.input_db), f"Database file {self.input_db} not found"

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def summarize_errors(model: str, client: GPTClient, raw_error_msg_list: list[str], full_code_list: list[str]) -> list[str]:

    system_prompt = f"""You are a FastAPI expert that helps summarize error messages and give concise guidance to avoid generating the same errors again. Your generated content will be directly fed to another coding LLM to avoid generating the same errors again.

Code Environment:
- Python version: {PYTHON_VERSION}
- FastAPI version compatible with Pydantic v2 (do NOT use v1-only features such as `orm_mode` in Config)
- Database is using SQLite3 with SQLAlchemy ORM

Your task:
1. Analyze the error message and the generated code
2. Identify the specific code snippets that caused the error
3. Provide a concise summary that includes:
   - The root cause of the error
   - The problematic code snippet(s) from the generated code
   - Clear guidance on how to fix/avoid this error

IMPORTANT: Keep your response under 1000 tokens. Focus on the most critical information.

You must directly respond plain text. Format your response as:
[Error Cause]: Brief description of what went wrong
[Problematic Code]:
```python
# The specific code that caused the error
```
[Guidance]: How to fix/avoid this error
"""
    
    def create_sum_request(raw_error_message: str, full_code: str):
        user_content = f"""Error Message:
{raw_error_message}

Generated Code:
```python
{full_code}
```

Please analyze the error and provide a summary with the problematic code snippet and guidance."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 16_000,
            "model": model,
        }

    logger.info(f"summarizing {len(raw_error_msg_list)} error messages...")
    requests = [create_sum_request(msg, code) for msg, code in zip(raw_error_msg_list, full_code_list)]
    responses = client.batch_chat_completion(requests, progress_bar=True)

    summaries = []

    for raw_msg, response in zip(raw_error_msg_list, responses):
        if not response:
            logger.error(f"failed to summarize error message: Empty response")
            summaries.append(raw_msg)  # Fallback to raw message
        else:
            logger.info(f"summarized error message: {len(response)} chars ({tools_token_count(response, model)} tokens)\nPreview: {response[:800]}...")
            summaries.append(response)
    
    return summaries
        




def load_existing_env_results(output_path: str) -> dict[str, dict]:
    """load existing environment results from output file for resume."""
    if not os.path.exists(output_path):
        return {}

    existing_map: dict[str, dict] = {}
    data = tools_jsonl_load(output_path)
    for item in data:
        scenario = normalize_scenario_name(item['scenario'])
        full_code = item.get('full_code', '')
        if full_code and isinstance(full_code, str) and len(full_code) > 10:
            existing_map[scenario] = item

    return existing_map


def test_run_specific_env(idx: int, env_config: dict) -> tuple[bool, str, dict]:
    import tempfile
    import uuid
    import random

    random.seed(idx)
    scenario_name = normalize_scenario_name(env_config["scenario"])
    unique_id = uuid.uuid4().hex[:8]
    
    port = get_random_available_port()
    wait_port_free(port)
    
    temp_dir = tempfile.mkdtemp(prefix=f"env_test_{unique_id}")
    temp_env_json = os.path.join(temp_dir, "test_env_code.jsonl")
    temp_db = os.path.join(temp_dir, "test_env.db")
    temp_server = os.path.join(temp_dir, "temp_server.py")
    
    logger.debug(f"{scenario_name} is testing on port {port}, temp_dir={temp_dir}")

    try:
        tools_jsonl_save([env_config], temp_env_json)

        shutil.copyfile(env_config["db_path"], temp_db)
        os.chmod(temp_db, 0o644)

        server_process = subprocess.Popen(
            [
                sys.executable, '-m', 'awm.core.server',
                '--port', str(port),
                '--scenario', scenario_name,
                '--db_path', temp_db,
                '--temp_server_path', temp_server,
                '--envs_load_path', temp_env_json
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True
        )

        time.sleep(3)

        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            logger.error(f"{scenario_name} Server process exited prematurely with code {server_process.returncode}")
            if stdout:
                output_preview = stdout[:2000] if len(stdout) > 2000 else stdout
                logger.error(f"{scenario_name} Server output:\n{'='*60}\n{output_preview}\n{'='*60}")
            return False, stdout, env_config
        

        if not wait_for_server(port, timeout=30):
            logger.error(f"{scenario_name} Server failed to start on port {port} (timeout after 30s)")
            
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                stdout, _ = server_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, _ = server_process.communicate()
            
            if stdout:
                output_preview = stdout[:2000] if len(stdout) > 2000 else stdout
                logger.error(f"{scenario_name} Server output:\n{'='*60}\n{output_preview}\n{'='*60}")
            else:
                logger.error(f"{scenario_name} Server produced no output")
            
            return False, stdout, env_config
        
        logger.info(f"{scenario_name} Server started successfully on port {port}")
        
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
            
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            server_process.wait()

        return True, "", env_config
    
    finally:
        # clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def batch_test_environments(env_configs: list[dict], max_workers: int = 8) -> list[tuple[bool, str, dict]]:
    if not env_configs:
        return []
    
    logger.info(f"testing {len(env_configs)} environments in parallel (max_workers={max_workers})...")
    
    results: list[tuple[bool, str, dict]] = [None] * len(env_configs)
    passed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {   
            executor.submit(test_run_specific_env, idx, config): idx
            for idx, config in enumerate(env_configs)
        }
        
        pbar = tqdm(total=len(env_configs), desc="Testing envs", unit="env")
        pbar.set_postfix(passed=passed, failed=failed)
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                scenario_name = env_configs[idx]['scenario']
                if result[0]:
                    passed += 1
                    logger.debug(f"{scenario_name}: test passed")
                else:
                    failed += 1
                    logger.debug(f"{scenario_name}: test failed due to {result}")
            except Exception as e:
                failed += 1
                logger.error(f"test execution failed for index {idx}: {e}")
                results[idx] = (False, str(e), env_configs[idx])
            
            pbar.set_postfix(passed=passed, failed=failed)
            pbar.update(1)
        
        pbar.close()
    
    logger.info(f"batch test done: {passed} passed, {failed} failed out of {len(env_configs)} total")
    return results


def test_all_environments(args: Config):
    env_configs = tools_jsonl_load(args.output)

    if args.allowed_scenarios is not None:
        env_configs = [e for e in env_configs if normalize_scenario_name(e["scenario"]) in args.allowed_scenarios]

    if not env_configs:
        logger.warning("No environments to test.")
        return

    logger.info(f"Test mode: loaded {len(env_configs)} environments from {args.output}")

    test_results = batch_test_environments(env_configs, max_workers=min(os.cpu_count() or 4, len(env_configs), 64))

    passed = []
    failed = []
    for env_config, (success, output, _) in zip(env_configs, test_results):
        scenario_name = env_config["scenario"]
        if success:
            passed.append(scenario_name)
        else:
            failed.append((scenario_name, output))

    for name in passed:
        logger.success(f"PASSED: {name}")

    
    logger.info(f"\n{'='*100}\n\n")

    for name, output in failed:
        error_preview = (output or "")[:500]
        logger.error(f"FAILED: {name}\n    {error_preview}")


    logger.info(f"\n{'='*100}\nTest Results: {len(passed)} passed, {len(failed)} failed out of {len(env_configs)} total.\nFailed={[item[0] for item in failed]}\n{'='*100}")

def generate_all_environments(args: Config):
    client = GPTClient(timeout=30 * 60)
    api_specs_data = tools_jsonl_load(args.input_spec)
    db_schemas_data = tools_jsonl_load(args.input_db)
    schema_map = {normalize_scenario_name(item["scenario"]): item["db_schema"] for item in db_schemas_data}
    
    processed_api_spec_data = []
    for api_spec_item in api_specs_data:
        scenario_name = normalize_scenario_name(api_spec_item["scenario"])

        if args.allowed_scenarios is not None and scenario_name not in args.allowed_scenarios:
            continue
        
        processed_api_spec_data.append(api_spec_item)

    # Resume: load and validate existing results
    existing_results = load_existing_env_results(args.output)
    results = []
    validated_scenarios: set[str] = set()

    if existing_results:
        target_scenarios = {normalize_scenario_name(item["scenario"]) for item in processed_api_spec_data}
        existing_to_validate = [config for scenario, config in existing_results.items() if scenario in target_scenarios]

        if existing_to_validate:
            logger.info(f"Resume: found {len(existing_to_validate)} existing results in {args.output}, validating...")
            test_results = batch_test_environments(
                existing_to_validate,
                max_workers=min(os.cpu_count() or 4, len(existing_to_validate), 64),
            )

            for env_config, (success, output, _) in zip(existing_to_validate, test_results):
                scenario = normalize_scenario_name(env_config['scenario'])
                if success:
                    validated_scenarios.add(scenario)
                    results.append(env_config)

            logger.info(
                f"resume validation: {len(validated_scenarios)}/{len(existing_to_validate)} passed, "
                f"will regenerate {len(existing_to_validate) - len(validated_scenarios)} failed scenarios"
            )

    items_to_generate = [item for item in processed_api_spec_data if normalize_scenario_name(item["scenario"]) not in validated_scenarios]

    if not items_to_generate:
        logger.info("all scenarios already validated via resume, nothing to generate.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        tools_jsonl_save(results, args.output)
        logger.info(f"total scenarios: {len(results)}/{len(processed_api_spec_data)}, saved to {args.output}")
        return results

    logger.info(f"generating {len(items_to_generate)}/{len(processed_api_spec_data)} scenarios (skipped {len(validated_scenarios)} validated)")

    def create_request(api_spec_item, error_summaries: list[str] | None = None):
        scenario_name = api_spec_item["scenario"]
        normalized_scenario_name = normalize_scenario_name(scenario_name)
        api_spec = api_spec_item["api_spec"]        
        db_schema = schema_map[normalized_scenario_name]
        
        api_spec_str = json.dumps(api_spec)
        db_schema_str = format_db_schema(db_schema)
        
        user_content = ENVIRONMENT_GENERATION_USER_PROMPT.format(
            PYTHON_VERSION=PYTHON_VERSION,
            scenario_name=scenario_name,
            api_spec=api_spec_str,
            database_schema=db_schema_str
        )

        if error_summaries:
            formatted_errors = []
            for i, summary in enumerate(error_summaries, 1):
                formatted_errors.append(f"Error #{i}:\n{summary}")
            all_errors = ("\n\n" + "="*50 + "\n\n").join(formatted_errors)
            user_content += f"\n\nAttention: You MUST avoid the following errors from previous attempts:\n{all_errors}"

        messages = [
            {
                "role": "system",
                "content": ENVIRONMENT_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,  # Lower temperature for more consistent code generation
            "max_tokens": 128_000,  # Increased for complex implementations
            "model": args.model,
        }
    
    current_items = items_to_generate
    current_requests = [create_request(item) for item in current_items]
    
    max_retries = args.max_retry

    # {scenario_name: [{"raw_error": str, "full_code": str, "summary": str}, ...]}
    history_error_messages: dict[str, list[dict[str, str]]] = {}

    for attempt in range(max_retries + 1):

        if not current_requests:
            break
            
        logger.info(f"Sending batch requests (Attempt {attempt+1}/{max_retries+1}) count={len(current_requests)}...")
        responses = client.batch_chat_completion(current_requests, progress_bar=True)
        
        candidates_to_test: list[tuple[dict, dict]] = []  # (api_spec_item, result)
        parse_failed: list[tuple[dict, str, str | None]] = []  # (api_spec_item, error_msg, full_code or None)
        
        for api_spec_item, response in zip(current_items, responses):
            scenario_name = api_spec_item["scenario"]
            error_msg = ""
            
            if not response:
                error_msg = "The generation returned an empty response. It might be due to a timeout (too long response) or triggering Azure OpenAI's safety filters."
                parse_failed.append((api_spec_item, error_msg, None))
            else:
                try:
                    logger.info(f"Processing response for {scenario_name}: {len(response)} chars")
                    env_spec = tools_robust_json_loads(response)
                    
                    db_filename = f"{normalize_scenario_name(scenario_name)}.db"
                    
                    if len(env_spec) == 0 or 'full_code' not in str(response)[:30]:
                        env_spec = {
                            'full_code': response
                        }

                    db_path = f"{args.database_dir}/{db_filename}"
                        
                    if "full_code" in env_spec:
                        result = {
                            "scenario": scenario_name,
                            "db_path": db_path,
                            "full_code": env_spec["full_code"]
                        }
                        logger.success(f"{scenario_name}: {len(env_spec['full_code'])} chars of code")
                        candidates_to_test.append((api_spec_item, result))

                    else:
                        error_msg = "The generated code violates the expected format. You must follow the specified format strictly."
                        logger.error(f"Failed to parse environment for {scenario_name}: {error_msg}")
                        parse_failed.append((api_spec_item, error_msg, response))

                except Exception as e:
                    error_msg = "The generated code violates the expected format. You must follow the specified format strictly."
                    logger.error(f"Failed to parse environment for {scenario_name}: {e}")
                    logger.error(f"Response preview: {response[:200]}...")
                    parse_failed.append((api_spec_item, error_msg, response))
        
        next_items = []
        pending_errors: list[tuple[dict, str, str]] = []
        
        if candidates_to_test:
            test_configs = [result for _, result in candidates_to_test]
            test_results = batch_test_environments(test_configs, max_workers=min(8, len(test_configs)))
            
            for (api_spec_item, result), (success, output, _) in zip(candidates_to_test, test_results):
                scenario_name = api_spec_item["scenario"]
                if success:
                    results.append(result)
                else:
                    error_msg = f"The generated code failed to start or crashed.\nOutput:\n{output}"
                    logger.error(f"Environment test run failed for {scenario_name}")
                    
                    full_code = result.get("full_code", "")
                    
                    if attempt < max_retries:
                        if scenario_name not in history_error_messages:
                            history_error_messages[scenario_name] = []
                        
                        existing_raw_errors = {e["raw_error"] for e in history_error_messages[scenario_name]}
                        if error_msg not in existing_raw_errors:
                            pending_errors.append((api_spec_item, error_msg, full_code))
                            next_items.append(api_spec_item)
                        else:
                            logger.warning(f"Duplicate error for {scenario_name}, using existing summaries...")
                            next_items.append(api_spec_item)
                    else:
                        logger.error(f"Given up on {scenario_name} after {max_retries+1} attempts, saving last attempt code.")
                        results.append(result)
        
        for api_spec_item, error_msg, full_code in parse_failed:
            scenario_name = api_spec_item["scenario"]
            if attempt < max_retries:
                if scenario_name not in history_error_messages:
                    history_error_messages[scenario_name] = []
                
                existing_raw_errors = {e["raw_error"] for e in history_error_messages[scenario_name]}
                if error_msg not in existing_raw_errors:
                    pending_errors.append((api_spec_item, error_msg, full_code or ""))
                    if api_spec_item not in next_items:
                        next_items.append(api_spec_item)
                else:
                    logger.warning(f"duplicate error for {scenario_name}, using existing summaries")
                    if api_spec_item not in next_items:
                        next_items.append(api_spec_item)
            else:
                last_code = full_code or ""
                if not last_code and scenario_name in history_error_messages:
                    for prev in reversed(history_error_messages[scenario_name]):
                        if prev["full_code"]:
                            last_code = prev["full_code"]
                            break
                if last_code:
                    db_filename = f"{normalize_scenario_name(scenario_name)}.db"
                    db_path = f"{args.database_dir}/{db_filename}"
                    results.append({
                        "scenario": scenario_name,
                        "db_path": db_path,
                        "full_code": last_code
                    })
                    logger.error(f"given up on {scenario_name} after {max_retries+1} attempts, saving last available code.")
                else:
                    logger.error(f"given up on {scenario_name} after {max_retries+1} attempts, no code available to save.")
        
        if pending_errors:
            raw_errors = [e[1] for e in pending_errors]
            full_codes = [e[2] for e in pending_errors]
            summarized_errors = summarize_errors(args.model, client, raw_errors, full_codes)
            
            for (api_spec_item, raw_error, full_code), summary in zip(pending_errors, summarized_errors):
                scenario_name = api_spec_item["scenario"]
                if scenario_name not in history_error_messages:
                    history_error_messages[scenario_name] = []
                history_error_messages[scenario_name].append({
                    "raw_error": raw_error,
                    "full_code": full_code,
                    "summary": summary
                })
        
        next_requests = []
        for api_spec_item in next_items:
            scenario_name = api_spec_item["scenario"]
            if scenario_name in history_error_messages:
                summaries = [e["summary"] for e in history_error_messages[scenario_name]]
                next_requests.append(create_request(api_spec_item, summaries))
            else:
                next_requests.append(create_request(api_spec_item, None))
        
        current_items = next_items
        current_requests = next_requests
    
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tools_jsonl_save(results, args.output)
    
    logger.info(f"total scenarios processed (generated environments): {len(results)}/{len(processed_api_spec_data)} (resumed={len(validated_scenarios)}, newly generated={len(results) - len(validated_scenarios)}), saved to {args.output}")


    avg_tokens_per_env = sum(tools_token_count(r['full_code'], args.model) for r in results) / len(results) if results else 0
    avg_chars_per_env = sum(len(r['full_code']) for r in results) / len(results) if results else 0
    avg_lines_per_env = sum(len(r['full_code'].splitlines()) for r in results) / len(results) if results else 0

    logger.info(f"average tokens per environment: {avg_tokens_per_env:.2f}, average chars per environment: {avg_chars_per_env:.2f}, average lines per environment: {avg_lines_per_env:.2f}")

    return results

    

def run(config: Config):
    generate_all_environments(config)

