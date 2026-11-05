"""Run real telecom episodes through the pinned AReaL Tau2Runner."""

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

from openai import AsyncOpenAI
import wandb


def load_runner(source):
    spec = importlib.util.spec_from_file_location('experiment_agent', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_episode(module, config, task_id, seed, output, model_limits):
    episode_id = f"task{config['task_ids'].index(task_id)}-seed{seed}"
    episode_dir = output / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    requests_file = (episode_dir / 'requests.jsonl').open('w')
    counters = {'agent_requests': 0, 'user_requests': 0,
                'user_tool_messages_before': 0, 'user_tool_messages_after': 0,
                'user_tool_calls_before': 0, 'user_tool_calls_after': 0}

    class RecordedRunner(module.Tau2Runner):
        def _make_completion_fn(self, client, time_list, is_agent=True):
            async def create(**kwargs):
                # Tau2's dummy pricing model is mapped to the real serving name here.
                kwargs['model'] = 'agent' if is_agent else 'user'
                kwargs['seed'] = seed
                counters['agent_requests' if is_agent else 'user_requests'] += 1
                messages = kwargs.get('messages', [])
                if not is_agent:
                    counters['user_tool_messages_after'] += sum(
                        message.get('role') == 'tool' for message in messages)
                    counters['user_tool_calls_after'] += sum(
                        len(message.get('tool_calls') or []) for message in messages)
                requests_file.write(json.dumps({'role': 'agent' if is_agent else 'user',
                                                'outbound': kwargs}, default=str) + '\n')
                requests_file.flush()
                async with model_limits['agent' if is_agent else 'user']:
                    response = await client.chat.completions.create(**kwargs)
                requests_file.write(json.dumps({'response': response.model_dump()}, default=str) + '\n')
                requests_file.flush()
                return response

            recorded_client = SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))
            completion = super()._make_completion_fn(recorded_client, time_list, is_agent)

            async def observed(*args, **kwargs):
                messages = deepcopy(kwargs.get('messages', []))
                if not is_agent:
                    counters['user_tool_messages_before'] += sum(
                        message.get('role') == 'tool' for message in messages)
                    counters['user_tool_calls_before'] += sum(
                        len(message.get('tool_calls') or []) for message in messages)
                requests_file.write(json.dumps({'role': 'agent' if is_agent else 'user',
                                                'incoming_messages': messages}, default=str) + '\n')
                requests_file.flush()
                return await completion(*args, **kwargs)

            return observed

        def _get_orchestrator(self, *args, **kwargs):
            orchestrator = super()._get_orchestrator(*args, **kwargs)
            original_run = orchestrator.arun

            async def observed_run():
                simulation = await original_run()
                (episode_dir / 'simulation.json').write_text(
                    simulation.model_dump_json(indent=2))
                return simulation

            orchestrator.arun = observed_run
            return orchestrator

    econfig = module.Tau2EnvConfig(
        domain='telecom', max_steps=config['max_steps'], solo_mode=False,
        user_llm='openai/user', user_llm_base_url=config['user_url'],
        user_llm_args={'temperature': 0.0, 'max_completion_tokens': 512,
                       'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
    started = time.monotonic()
    async with AsyncOpenAI(base_url=config['agent_url'], api_key='local', max_retries=0,
                           timeout=120) as agent_client, AsyncOpenAI(
            base_url=config['user_url'], api_key='local', max_retries=0,
            timeout=120) as user_client:
        runner = RecordedRunner(econfig, {'temperature': 0.6,
                                         'max_completion_tokens': config.get('agent_max_tokens', 1024)},
                                agent_client, user_client)
        task = module._get_task('telecom', task_id, config['split'])
        try:
            info = await asyncio.wait_for(runner.run(task), timeout=config['episode_timeout'])
            (episode_dir / 'trajectory.json').write_text(info.model_dump_json(indent=2))
            result = {'episode': episode_id, 'task_id': task_id, 'seed': seed,
                      'reward': info.reward, 'error': info.error,
                      'agent_turns': sum(message.role == 'assistant' for message in info.messages),
                      'user_turns': sum(message.role == 'user' for message in info.messages)}
        except Exception as error:
            result = {'episode': episode_id, 'task_id': task_id, 'seed': seed,
                      'reward': None, 'error': f'{type(error).__name__}: {error}'}
    requests_file.close()
    result.update(counters)
    result['elapsed_seconds'] = time.monotonic() - started
    (episode_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


async def main(args):
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2))
    module = load_runner(args.source)
    run = wandb.init(project='rlbench-tau2', mode='offline', dir=str(output), config=config)
    semaphore = asyncio.Semaphore(config['concurrency'])
    model_limits = {role: asyncio.Semaphore(config.get('requests_per_model', 1))
                    for role in ('agent', 'user')}

    async def bounded(task_id, seed):
        async with semaphore:
            result = await run_episode(module, config, task_id, seed, output, model_limits)
            run.log({key: value for key, value in result.items()
                     if isinstance(value, (int, float))})
            return result

    results = await asyncio.gather(*(bounded(task_id, seed)
                                   for seed in config['seeds']
                                   for task_id in config['task_ids']))
    valid = [item for item in results if item['error'] is None]
    summary = {'episodes': len(results), 'valid_episodes': len(valid),
               'errors': len(results) - len(valid),
               'mean_reward': sum(item['reward'] for item in valid) / len(valid) if valid else None,
               'user_tool_messages_before': sum(item['user_tool_messages_before'] for item in results),
               'user_tool_messages_after': sum(item['user_tool_messages_after'] for item in results),
               'user_tool_calls_before': sum(item['user_tool_calls_before'] for item in results),
               'user_tool_calls_after': sum(item['user_tool_calls_after'] for item in results)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    run.summary.update(summary)
    run.finish()
    print(json.dumps(summary), flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    asyncio.run(main(parser.parse_args()))
