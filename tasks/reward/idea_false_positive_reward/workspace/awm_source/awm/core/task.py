import os
from awm.gpt import GPTClient
from awm.prompts import TASK_GENERATION_SYSTEM_PROMPT, TASK_GENERATION_USER_PROMPT
from awm.tools import tools_robust_json_loads, tools_jsonl_save, tools_jsonl_load
from loguru import logger
from dataclasses import dataclass
import random



@dataclass
class Config:
    input: str # scenario description file
    output: str
    num_tasks: int = 10
    shuffle: bool = True
    limit: int | None = None
    model: str = "your-llm-model-name"
    max_retry: int = 4  # Maximum number of retry attempts

    def pre_process(self):
        if self.limit is None: self.limit = int(1e18)
        
        assert self.input is not None and os.path.exists(self.input), "input is required"
        
        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def generate_all_tasks(args: Config, scenarios: list[dict]):
    logger.info(f"Preparing batch requests for {len(scenarios)} scenarios...")
    
    def create_request(scenario):
        messages = [
            {
                "role": "system",
                "content": TASK_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": TASK_GENERATION_USER_PROMPT.format(
                    scenario_name=scenario["name"],
                    scenario_description=scenario["description"],
                    num_tasks=args.num_tasks,
                )
            }
        ]
        
        return {
            "messages": messages,
            "temperature": 1.0,
            "model": args.model,
            "max_tokens": 32_000,
        }
    
    # Initial batch
    current_items = scenarios
    current_requests = [create_request(scenario) for scenario in current_items]
    
    results = []
    client = GPTClient()
    max_retries = args.max_retry

    for attempt in range(max_retries + 1):
        if not current_requests:
            break
        
        logger.info(f"Sending batch requests (Attempt {attempt+1}/{max_retries+1}) count={len(current_requests)}...")
        responses = client.batch_chat_completion(current_requests, progress_bar=True)
        
        next_items = []
        next_requests = []
        
        for scenario, response in zip(current_items, responses):
            need_retry = False
            
            if not response:
                logger.error(f"Failed to generate tasks for {scenario['name']}: Empty response")
                need_retry = True
            else:
                try:
                    logger.info(f"Processing response for {scenario['name']}: {len(response)} chars")
                    tasks = tools_robust_json_loads(response)
                    
                    task_list = tasks.get('tasks', [])
                    
                    # Check if we got the required number of tasks
                    if len(task_list) < args.num_tasks:
                        logger.warning(f"{scenario['name']}: Only got {len(task_list)}/{args.num_tasks} tasks, scheduling retry...")
                        need_retry = True
                    else:
                        results.append({
                            "scenario": scenario["name"],
                            "tasks": task_list[:args.num_tasks],
                        })
                        logger.success(f"✓ {scenario['name']}: {len(task_list)} tasks generated")
                    
                except Exception as e:
                    logger.error(f"Failed to parse tasks for {scenario['name']}: {e}")
                    logger.error(f"Response preview: {response[:200]}...")
                    need_retry = True
            
            # Schedule retry if needed
            if need_retry and attempt < max_retries:
                logger.warning(f"Scheduling retry for {scenario['name']}...")
                next_items.append(scenario)
                next_requests.append(create_request(scenario))
            elif need_retry:
                logger.error(f"Given up on {scenario['name']} after {max_retries+1} attempts.")
        
        current_items = next_items
        current_requests = next_requests
    
    logger.info(f"Total scenarios processed: {len(results)}/{len(scenarios)}")
    return results


    


def run(config: Config):
    logger.info(f"Generating tasks for all scenarios...")

    scenarios = tools_jsonl_load(config.input)
    if config.shuffle:
        random.shuffle(scenarios)
    scenarios = scenarios[:config.limit]

    results = generate_all_tasks(config, scenarios)

    logger.info("\n" + "="*80)
    for result in results:
        logger.info(f"\n{result['scenario']}:")
        logger.info("-" * 80)
        for i, task in enumerate(result['tasks'], 1):
            if i >= 3: break
            logger.info(f"{i}. {task}")

    os.makedirs(os.path.dirname(config.output), exist_ok=True)
    tools_jsonl_save(results, config.output)

    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to {config.output}")
