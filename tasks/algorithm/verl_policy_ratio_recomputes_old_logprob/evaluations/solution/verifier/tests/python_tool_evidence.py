"""Validate Python-tool evidence as data; never execute submitted code."""
from collections import Counter
import hashlib
import json
import re


def evaluate_python_tools(tools, data_root, expected_count, training_samples=None):
    sources = {}
    for split in ['train', 'test']:
        for line in (data_root/(split+'.jsonl')).read_text().splitlines():
            row = json.loads(line)
            sources[split, row['extra_info']['index']] = row['prompt'][0]['content']
    assert len(tools) == expected_count, 'expected exactly one Python call per input prompt'
    test_indices = {index for split, index in sources if split == 'test'}
    assert Counter(row['index'] for row in tools if row['split'] == 'test') == Counter({index: 4 for index in test_indices}), 'missing per-question held-out Python calls'
    if training_samples is not None:
        train_calls = Counter(row['index'] for row in tools if row['split'] == 'train')
        assert Counter({index: count*4 for index, count in train_calls.items()}) == Counter(row['index'] for row in training_samples), 'training Python calls do not match generated answer groups'
    failures = successes = errors = 0
    for row in tools:
        assert row['tool'] == 'code_interpreter', 'wrong tool'
        assert row['user_prompt'] == sources[row['split'], row['index']], 'Python input uses a different problem'
        raw = row['raw_code_response']
        match = re.search(r'```(?:python)?\s*\n(.*?)```', raw, re.DOTALL)
        assert row['code'] == (match[1].strip() if match else raw.strip()), 'code differs from model output'
        assert row['code_sha256'] == hashlib.sha256(row['code'].encode()).hexdigest(), 'code hash mismatch'
        command = row['command']
        assert command[-1] == row['code'], 'interpreter did not receive generated code'
        assert command[1:-1] in [['-I', '-S'], ['-I', '-S', '-c']], 'unexpected Python invocation'
        has_c = command[1:-1] == ['-I', '-S', '-c']
        assert row['invocation_has_c'] == has_c, 'incorrect invocation observation'
        failures += not has_c
        assert isinstance(row['returncode'], int), 'missing real process exit status'
        observation = json.loads(row['observation'])
        assert observation == {key: row[key] for key in ['returncode', 'stdout', 'stderr', 'timed_out']}, 'tool result changed before continuation'
        assert row['observation'] in row['continuation_prompt'], 'model did not receive the interpreter observation'
        assert row['user_prompt'] in row['continuation_prompt'], 'continuation lost original problem'
        assert json.dumps({'name': 'code_interpreter', 'arguments': {'code': row['code']}}) in row['continuation_prompt'], 'continuation lost Python call'
        successes += row['returncode'] == 0 and bool(row['stdout'].strip())
        errors += row['returncode'] != 0
    result = {'tool_requests': len(tools), 'tool_invocation_failures': failures,
              'python_successful_outputs': successes, 'python_execution_errors': errors}
    assert failures == 0, f'{failures} Python calls omit -c and treat generated code as a filename'
    assert successes > 0, 'no generated Python code produced a successful nonempty result'
    return result
