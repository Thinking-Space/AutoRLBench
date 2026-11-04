import json
from awm.gpt import GPTClient
from awm.prompts import API_GENERATION_SYSTEM_PROMPT, API_GENERATION_USER_PROMPT
from awm.tools import tools_robust_json_loads, tools_jsonl_load, tools_jsonl_save, format_db_schema, normalize_scenario_name
from loguru import logger
from dataclasses import dataclass
import os

@dataclass
class Config:
    input_task: str
    input_db: str
    output: str
    model: str = "your-llm-model-name"
    max_retry: int = 4  # Maximum number of retry attempts

    def pre_process(self):
        assert os.path.exists(self.input_task), f"Task file {self.input_task} not found"
        assert os.path.exists(self.input_db), f"Database file {self.input_db} not found"
        assert self.output is not None and self.output.endswith('.jsonl'), "Output path .jsonl is required"

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def generate_all_api_specs(args: Config):
    tasks_data = tools_jsonl_load(args.input_task)
    schema_data = tools_jsonl_load(args.input_db)    
    schema_map = {normalize_scenario_name(item["scenario"]): item for item in schema_data}

    def create_request(scenario_data):
        scenario_name = scenario_data["scenario"]
        schema_item = schema_map[normalize_scenario_name(scenario_name)]
        tasks_list = "\n".join([f"- {task}" for task in scenario_data["tasks"]])
        schema_str = format_db_schema(schema_item["db_schema"])
        
        messages = [
            {
                "role": "system",
                "content": API_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": API_GENERATION_USER_PROMPT.format(
                    scenario_name=scenario_name,
                    tasks_list=tasks_list,
                    database_schema=schema_str
                )
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
    client = GPTClient(timeout=30*60)
    max_retries = args.max_retry

    for attempt in range(max_retries + 1):
        if not current_requests:
            break
        
        logger.info(f"Sending batch requests (Attempt {attempt+1}/{max_retries+1}) count={len(current_requests)}...")
        responses = client.batch_chat_completion(current_requests, progress_bar=True)
        
        next_items = []
        next_requests = []

        for scenario_data, response in zip(current_items, responses):
            scenario_name = scenario_data["scenario"]
            need_retry = False
            
            if not response:
                logger.error(f"Failed to generate API spec for {scenario_name}: Empty response")
                need_retry = True
            else:
                try:
                    api_spec = tools_robust_json_loads(response)
                    
                    api_groups = api_spec.get("api_groups", [])
                    total_endpoints = sum(len(g.get("endpoints", [])) for g in api_groups)
                    
                    # Validate API spec structure
                    if not api_groups or total_endpoints == 0:
                        logger.error(f"Empty API spec for {scenario_name}")
                        need_retry = True
                    else:
                        results.append({
                            "scenario": scenario_name,
                            "api_spec": api_spec
                        })
                        logger.success(f"{scenario_name}: {len(api_groups)} groups, {total_endpoints} endpoints")
                        continue
                    
                except Exception as e:
                    logger.error(f"Failed to parse API spec for {scenario_name}: {e}")
                    logger.error(f"Response preview: {response[:200]}...")
                    need_retry = True
            
            if need_retry and attempt < max_retries:
                logger.warning(f"Scheduling retry for {scenario_name}...")
                next_items.append(scenario_data)
                next_requests.append(create_request(scenario_data))
            elif need_retry:
                logger.error(f"Given up on {scenario_name} after {max_retries+1} attempts.")
        
        current_items = next_items
        current_requests = next_requests
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tools_jsonl_save(results, args.output)
    logger.info(f"Total scenarios processed: {len(results)}/{len(tasks_data)}, generated API Specs to {args.output}")

    return results

def run(config: Config):
    generate_all_api_specs(config)
