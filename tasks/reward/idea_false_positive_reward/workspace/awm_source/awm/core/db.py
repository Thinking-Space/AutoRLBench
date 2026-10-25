import sqlite3
import os
from pathlib import Path
from awm.gpt import GPTClient
from awm.prompts import DATABASE_GENERATION_SYSTEM_PROMPT, DATABASE_GENERATION_USER_PROMPT
from loguru import logger
from dataclasses import dataclass
from awm.tools import tools_jsonl_save, tools_jsonl_load, tools_robust_json_loads, normalize_scenario_name, tools_token_count

@dataclass
class Config:
    input: str # task file
    output: str # generated database schema file
    database_dir: str = './outputs/databases'
    model: str = ""
    allowed_scenarios: list[str] | None = None  # Comma-separated list of allowed scenarios
    max_retry: int = 4  # Maximum number of retry attempts
    error_threshold: float = 0.1  # If more than this ratio of tables fail, retry
    

    def pre_process(self):
        assert os.path.exists(self.input), f"Task file {self.input} not found"
        assert self.output is not None and self.output.endswith('.jsonl'), "Output path .jsonl is required"
        if self.allowed_scenarios:
            self.allowed_scenarios = {normalize_scenario_name(name) for name in self.allowed_scenarios}


def summarize_errors(args: Config, client: GPTClient, raw_error_msg_list: list[str]) -> list[str]:
    
    def create_sum_request(raw_error_message: str):
        messages = [
            {
                "role": "system",
                "content": "You are an SQL expert that helps summarize SQL errors during creating a SQLite3 database  and give concise guidance to avoid generating the same errors again. Your generated content will be directly fed to another coding LLM to avoid generating the same errors again. You must directly respond plain text when user gives you the error messages."
            },
            {
                "role": "user",
                "content": raw_error_message
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 10_000,
            "model": args.model,
        }

    logger.info(f"Summarizing {len(raw_error_msg_list)} error messages...")
    requests = [create_sum_request(msg) for msg in raw_error_msg_list]
    responses = client.batch_chat_completion(requests, progress_bar=True)

    summaries = []

    for raw_msg, response in zip(raw_error_msg_list, responses):
        if not response:
            logger.error(f"Failed to summarize error message: Empty response")
            summaries.append(raw_msg)  # Fallback to raw message
        else:
            logger.info(f"Summarized error message: {len(response)} chars ({tools_token_count(response, args.model)} tokens)\nPreview: {response[:800]}...")
            summaries.append(response)
    
    return summaries


def create_sqlite_database(scenario_name: str, db_schema: dict, db_dir: str) -> tuple[str, int, int, set[str]]:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    
    db_filename = f"{normalize_scenario_name(scenario_name)}.db"
    db_path = os.path.join(db_dir, db_filename)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.debug(f"Removed existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info(f"Creating database for {scenario_name}...")
    
    successful_tables = 0
    failed_tables = 0
    error_messages: set[str] = set()
    
    for table in db_schema.get("tables", []):
        table_name = table.get('name', 'unknown')
        try:
            cursor.execute(table["ddl"])
            
            for index_sql in table.get("indexes", []):
                cursor.execute(index_sql)
            
            for example_sql in table.get("examples", []):
                cursor.execute(example_sql)
            
            successful_tables += 1
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            failed_tables += 1
            error_messages.add(f"Table {table_name}: {str(e)[:3000]}")
    
    conn.commit()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    logger.success(f"Database created with {len(tables)} tables: {[t[0] for t in tables]}")
    
    conn.close()
    
    return db_path, successful_tables, failed_tables, error_messages


def generate_all_databases(args: Config):
    user_intentions = tools_jsonl_load(args.input)
    
    logger.info(f"Preparing batch requests for {len(user_intentions)} scenarios...")
    
    # Filter scenarios
    requested_intentions = []
    for data in user_intentions:
        scenario_name = normalize_scenario_name(data["scenario"])
        if args.allowed_scenarios is not None and scenario_name not in args.allowed_scenarios:
            continue
        requested_intentions.append(data)
    
    def create_request(data, error_msg=None):
        scenario_name = data["scenario"]
        user_intentions_str = '\n'.join([f"- {intention}" for intention in data["tasks"]])
        
        user_content = DATABASE_GENERATION_USER_PROMPT.format(
            scenario_name=scenario_name,
            user_intentions=user_intentions_str,
            num_tasks=len(data["tasks"]),
        )
        
        if error_msg and error_msg != "":
            user_content += f"\n\nAttention: You MUST avoid generating SQL codes that caused the following errors in previous attempts. You MUST refer the below guidelines to help you avoid generating errors:\n{error_msg}"
        
        messages = [
            {
                "role": "system",
                "content": DATABASE_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 128_000,
            "model": args.model,
        }
    
    # Initial batch
    current_items = requested_intentions
    current_requests = [create_request(item) for item in current_items]
    
    results = []
    client = GPTClient(timeout=600)
    max_retries = args.max_retry
    error_threshold = args.error_threshold
    
    # Track history for each scenario
    history_error_messages: dict[str, set[str]] = {}
    history_data_messages: dict[str, list[tuple[float, dict]]] = {}

    for attempt in range(max_retries + 1):
        if not current_requests:
            break
        
        logger.info(f"Sending batch requests (Attempt {attempt+1}/{max_retries+1}) count={len(current_requests)}...")
        responses = client.batch_chat_completion(current_requests, progress_bar=True)
        
        next_items = []
        next_requests = []

        for data, response in zip(current_items, responses):
            scenario_name = data["scenario"]
            error_msg: set[str] = set()
            db_schema = {}
            error_ratio = 1.0
            
            if not response:
                logger.error(f"Failed to generate database schema for {scenario_name}: Empty response")
            else:
                try:
                    logger.info(f"Processing response for {scenario_name}: {len(response)} chars")
                    db_schema = tools_robust_json_loads(response)
                    
                    db_data = {
                        "scenario": scenario_name,
                        "db_schema": db_schema,
                        "db_path": None
                    }
                    
                    db_path, successful, failed, error_msg = create_sqlite_database(
                        scenario_name, db_schema, args.database_dir
                    )
                    db_data['db_path'] = db_path
                    
                    total_tables = successful + failed
                    if total_tables > 0:
                        error_ratio = failed / total_tables
                    else:
                        error_ratio = 1.0
                    
                    tables = db_schema.get("tables", [])
                    logger.info(f"{scenario_name}: {len(tables)} tables, error ratio: {error_ratio:.2%}")
                    
                    if error_ratio > error_threshold:
                        logger.warning(f"Error ratio too high for {scenario_name}: {failed} / {total_tables} = {error_ratio:.2%} > {error_threshold}, scheduling retry...")
                    else:
                        results.append(db_data)
                        logger.success(f"{scenario_name}: Successfully processed with {error_ratio:.2%} error rate")
                        continue
                    
                except Exception as e:
                    logger.error(f"Failed to generate/create database for {scenario_name}: {e}")
                    logger.error(f"Response preview: {response[:200]}...")
            
            temp = normalize_scenario_name(scenario_name)
            if temp not in history_data_messages:
                history_data_messages[temp] = []
            
            failed_result = {
                "scenario": scenario_name,
                "db_schema": db_schema,
                "db_path": f"{args.database_dir}/{normalize_scenario_name(scenario_name)}.db"
            }
            history_data_messages[temp].append((error_ratio, failed_result))
            
            if attempt < max_retries:
                if temp not in history_error_messages:
                    history_error_messages[temp] = set()
                history_error_messages[temp] = history_error_messages[temp].union(error_msg)
                
                combined_error_msg = f"\n{'='*80}\n".join(list(history_error_messages[temp]))
                if len(combined_error_msg) > 64_000:
                    combined_error_msg = combined_error_msg[-64_000:] + "... (truncated)"
                
                next_items.append(data)
                next_requests.append((data, combined_error_msg))
            else:
                min_error, min_error_result = sorted(history_data_messages[temp], key=lambda x: x[0])[0]
                results.append(min_error_result)
                logger.error(f"Given up on {scenario_name} after {max_retries+1} attempts. Selecting best result with error_ratio={min_error:.2%}")
        
        if next_requests:
            summarized_errors = summarize_errors(args, client, [req[1] for req in next_requests])
            next_requests = [create_request(req[0], err_msg) for req, err_msg in zip(next_requests, summarized_errors)]
        
        current_items = next_items
        current_requests = next_requests

    tools_jsonl_save(results, args.output)

    logger.info(f"\nDatabase schemas saved to {args.output}")
    logger.info(f"Total scenarios processed: {len(results)}/{len(requested_intentions)}")
    return results



def run(config: Config):
    generate_all_databases(config)
