import argparse
import asyncio
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion


async def check(source):
    spec = importlib.util.spec_from_file_location('checked_agent', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []

    async def create(**kwargs):
        received.append(kwargs['messages'])
        return ChatCompletion(id='contract', object='chat.completion', created=0,
                              model='user', choices=[{'index': 0, 'finish_reason': 'stop',
                              'message': {'role': 'assistant', 'content': 'ok'}}])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    runner = module.Tau2Runner(module.Tau2EnvConfig(), {}, client, client)
    cases = [[], [{'role': 'user', 'content': 'Hello', 'tool_calls': None}], [
        {'role': 'system', 'content': 'You are the customer.'},
        {'role': 'user', 'content': 'Check the signal.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'check_status_bar', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'No Service'},
    ], [
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'check_status_bar', 'arguments': '{}'}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'check_network_status', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'No Service'},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'Airplane mode enabled'},
        {'role': 'assistant', 'content': 'I checked both.', 'tool_calls': []},
    ]]
    results = []
    for is_agent in (False, True):
        for index, messages in enumerate(cases):
            original = deepcopy(messages)
            expected = [{key: value for key, value in message.items()
                         if key != 'tool_calls' or value is not None} for message in messages]
            cleaner_kwargs = {}
            if 'for_user' in inspect.signature(runner._clean_messages).parameters:
                cleaner_kwargs['for_user'] = not is_agent
            direct = runner._clean_messages(messages, **cleaner_kwargs)
            results.append({'case': index, 'is_agent': is_agent, 'entry': 'cleaner',
                            'preserved': direct == expected,
                            'input_unchanged': messages == original,
                            'input_count': len(messages), 'output_count': len(direct)})
            completion = runner._make_completion_fn(client, [], is_agent=is_agent)
            await completion(model='user', messages=messages)
            results.append({'case': index, 'is_agent': is_agent, 'entry': 'completion',
                            'preserved': received[-1] == expected,
                            'input_unchanged': messages == original,
                            'input_count': len(messages), 'output_count': len(received[-1])})
    return {'checks': results, 'passed': all(item['preserved'] and item['input_unchanged']
                                           for item in results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = asyncio.run(check(args.source))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
