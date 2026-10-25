import sqlite3
from pathlib import Path
from awm.gpt import GPTClient
from awm.prompts import SAMPLE_DATA_GENERATION_SYSTEM_PROMPT, SAMPLE_DATA_GENERATION_USER_PROMPT
from awm.tools import tools_robust_json_loads, tools_jsonl_load, tools_jsonl_save, normalize_scenario_name, format_db_schema, tools_token_count
from loguru import logger
from dataclasses import dataclass
from awm.core.db import create_sqlite_database
import os

@dataclass
class Config:
    input_task: str
    input_db: str
    output: str
    database_dir: str = './outputs/databases'
    model: str = "your-llm-model-name"
    allowed_scenarios: list[str] | None = None  # Comma-separated list of allowed scenarios
    max_retry: int = 4  # Maximum number of retry attempts
    error_threshold: float = 0.1  # If more than this ratio of statements fail, retry


    def pre_process(self):
        assert os.path.exists(self.input_task), f"Task file {self.input_task} not found"
        assert os.path.exists(self.input_db), f"Database file {self.input_db} not found"
        assert self.output.endswith('.jsonl'), "Output path .jsonl is required"

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def summarize_errors(args: Config, client: GPTClient, raw_error_msg_list: list[str]) -> list[str]:

    def create_sum_request(raw_error_message: str):
        messages = [
            {
                "role": "system",
                "content": "You are an SQL expert that helps summarize SQL error messages and give concise guidance to to avoid generating the same errors again. Your generated content will be directly fed to another coding LLM to avoid generating the same errors again. You must directly respond plain text when user gives you the error messages."
            },
            {
                "role": "user",
                "content": raw_error_message
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 8_000,
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


def execute_sample_data(db_path: str, sample_data: dict, scenario_name: str) -> tuple[int, int, set[str]]:
    
    db_file = Path(db_path)
    if not db_file.exists():
        logger.error(f"Database not found: {db_path}, skipping data insertion")
        return 0, 0, {"Database file not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    total_statements = 0
    successful_inserts = 0
    failed_inserts = 0
    error_messages: set[str] = set()
    
    try:
        for table_data in sample_data.get("tables", []):
            table_name = table_data.get("table_name", "unknown")
            insert_statements = table_data.get("insert_statements", [])
            
            
            for stmt in insert_statements:
                total_statements += 1
                try:
                    cursor.execute(stmt)
                    successful_inserts += 1
                except sqlite3.IntegrityError as e:
                    failed_inserts += 1
                    error_messages.add(f"Table {table_name} - IntegrityError: {str(e)[:1000]}")
                except sqlite3.Error as e:
                    failed_inserts += 1
                    error_messages.add(f"Table {table_name} - SQLError: {str(e)[:1000]}\nStatement: {stmt[:1000]}")
        
        conn.commit()
        logger.info(f"{scenario_name}: {successful_inserts}/{total_statements} statements executed with {failed_inserts} failures")
        
    except Exception as e:
        conn.rollback()
        error_messages.add(f"Transaction failed: {str(e)}")
        raise e
    finally:
        conn.close()
    
    return successful_inserts, failed_inserts, error_messages


def generate_and_insert_sample_data(args: Config):
    
    tasks_data = tools_jsonl_load(args.input_task)
    db_schemas_data = tools_jsonl_load(args.input_db)
    db_schemas_data = {normalize_scenario_name(item["scenario"]): item for item in db_schemas_data}
    
    logger.info(f"Preparing batch requests for {len(tasks_data)} scenarios...")
    
    def create_request(task_item, error_msg=None):
        scenario_name = task_item["scenario"]
        schema_item = db_schemas_data[normalize_scenario_name(scenario_name)]
        tasks_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(task_item["tasks"])])
        schema_str = format_db_schema(schema_item["db_schema"])
        
        user_content = SAMPLE_DATA_GENERATION_USER_PROMPT.format(
            scenario_name=scenario_name,
            tasks_list=tasks_list,
            database_schema=schema_str
        )
        
        if error_msg and error_msg != "":
            user_content += f"\n\nAttention: You MUST avoid generating SQL codes that caused the following errors in previous attempts. You MUST refer the below guidelines to help you avoid generating errors:\n{error_msg}"
        
        messages = [
            {
                "role": "system",
                "content": SAMPLE_DATA_GENERATION_SYSTEM_PROMPT
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
    
    current_items = tasks_data
    current_requests = [create_request(item) for item in current_items]
    
    results = []
    client = GPTClient()
    max_retries = args.max_retry
    error_threshold = args.error_threshold
    
    history_error_messages = {}
    history_data_messages = {}
    total_successful = 0
    total_failed = 0

    for attempt in range(max_retries + 1):
        if not current_requests:
            break
        
        logger.info(f"Sending batch requests (Attempt {attempt+1}/{max_retries+1}) count={len(current_requests)}...")
        responses = client.batch_chat_completion(current_requests, progress_bar=True)
        
        next_items = []
        next_requests = []

        for task_item, response in zip(current_items, responses):
            scenario_name = task_item["scenario"]
            error_msg = set()
            sample_data = {}
            total_tables = 0
            total_inserts = 0
            error_ratio = 1.0
            
            if not response:
                logger.error(f"Failed to generate sample data for {scenario_name}: Empty response")
            else:
                try:
                    logger.info(f"Processing response for {scenario_name}: {len(response)} chars ({tools_token_count(response, args.model)} tokens)")
                    sample_data = tools_robust_json_loads(response)
                    
                    total_tables = len(sample_data.get("tables", []))
                    total_inserts = sum(
                        len(table.get("insert_statements", [])) 
                        for table in sample_data.get("tables", [])
                    )
                    
                    logger.info(f"Generated {total_tables} tables with {total_inserts} INSERT statements")
                    
                    db_filename = f"{normalize_scenario_name(scenario_name)}.db"
                    db_path = f"{args.database_dir}/{db_filename}"
                    
                    # reset the database before inserting new data
                    create_sqlite_database(scenario_name, db_schemas_data[normalize_scenario_name(scenario_name)]["db_schema"], args.database_dir)
                    successful, failed, error_msg = execute_sample_data(db_path, sample_data, scenario_name)
                    
                    total_statements = successful + failed
                    if total_statements > 0:
                        error_ratio = failed / total_statements
                    else:
                        error_ratio = 1.0
                    
                    if error_ratio > error_threshold:
                        logger.warning(f"Error ratio too high for {scenario_name}: {failed} / {total_statements} ={error_ratio:.2%} > {error_threshold}, scheduling retry...")
                    else:
                        result = {
                            "scenario": scenario_name,
                            "tables_count": total_tables,
                            "inserts_count": total_inserts,
                            "sample_data": sample_data
                        }
                        results.append(result)
                        total_successful += successful
                        total_failed += failed
                        logger.success(f"{scenario_name}: Successfully processed with {error_ratio:.2%} error rate")
                        continue
                    
                except Exception as e:
                    error_msg = f"Failed to parse or process sample data: {str(e)}"
                    logger.error(f"Failed to process sample data for {scenario_name}: {e}")
                    logger.error(f"Response preview: {response[:200]}...")
                    import traceback
                    logger.error(traceback.format_exc())
            
            
            temp = normalize_scenario_name(scenario_name)
            if temp not in history_data_messages:
                history_data_messages[temp] = []
            
            failed_result = {
                "scenario": scenario_name,
                "tables_count": total_tables,
                "inserts_count": total_inserts,
                "sample_data": sample_data,
            }
            history_data_messages[temp].append(
                (error_ratio, failed_result)
            )

            if attempt < max_retries:
                
                if temp not in history_error_messages:
                    history_error_messages[temp] = set()
                history_error_messages[temp] = history_error_messages[temp].union(error_msg)

                combined_error_msg = f"\n{'='*80}\n".join(list(history_error_messages[temp]))
                if len(combined_error_msg) > 64_000:
                    combined_error_msg = combined_error_msg[-64_000:] + "... (truncated)"

                next_items.append(task_item)
                next_requests.append((task_item, combined_error_msg))
            else:
                min_error, min_error_result = sorted(history_data_messages[temp], key=lambda x: x[0])[0]
                results.append(min_error_result)
                logger.error(f"Given up on {scenario_name} after {max_retries+1} attempts. Selecting best result with {min_error_result['inserts_count']} inserts. error_ratio={min_error}")
        

        summarized_errors = summarize_errors(args, client, [req[1] for req in next_requests])
        next_requests = [create_request(req[0], err_msg) for req, err_msg in zip(next_requests, summarized_errors)]
        current_items = next_items
        current_requests = next_requests
    

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tools_jsonl_save(results, args.output)
    
    logger.success(f"\n{'='*100}")
    logger.success(f"Sample data generation complete!")
    logger.info(f"Scenarios processed: {len(results)}/{len(tasks_data)}")
    logger.info(f"Total INSERT statements executed: {total_successful}")
    
    return results


def run(config: Config):
    generate_and_insert_sample_data(config)
