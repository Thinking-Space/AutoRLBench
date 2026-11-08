"""CPU input schema used by the rerun and the independent verifier."""
import hashlib
import json
from pathlib import Path
import re

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular(path, root):
    path, root = Path(path), Path(root)
    relative = path.relative_to(root)
    current = root
    if current.is_symlink():
        raise ValueError('Symlink root is not accepted')
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlinks are not accepted: ' + str(relative))
    if not path.is_file():
        raise ValueError('Missing regular file: ' + str(relative))
    return path


def read_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def gold(answer):
    found = re.search(r'#### (\-?[0-9\.\,]+)', answer)
    if found is None:
        raise ValueError('Invalid raw answer format')
    return found.group(1).replace(',', '')


def prompt(question):
    return [{'role': 'user', 'content': question + ' ' + INSTRUCTION}]


def check_rows(raw, actual, split):
    if len(raw) != len(actual):
        raise ValueError('Row count changed: ' + split)
    for index, (source, row) in enumerate(zip(raw, actual, strict=True)):
        if row.get('prompt') != prompt(source['question']):
            raise ValueError(f'Policy prompt differs from the original question and instruction: {split}[{index}]')
        if row.get('reward_model') != {'style': 'rule', 'ground_truth': gold(source['answer'])}:
            raise ValueError(f'Reward label changed: {split}[{index}]')
        if row.get('data_source') != 'openai/gsm8k' or row.get('ability') != 'math':
            raise ValueError('Data identity changed')
        extra = row.get('extra_info', {})
        if extra.get('split') != split or extra.get('index') != index:
            raise ValueError('Row order or identity changed')
        if extra.get('question') != source['question'] or extra.get('answer') != source['answer']:
            raise ValueError('Source metadata changed')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_pins(workspace, specification):
    workspace = Path(workspace)
    for relative, expected in specification['pinned_workspace'].items():
        path = regular(workspace / relative, workspace)
        if digest(path) != expected:
            raise ValueError('Non-target input or code changed: ' + relative)
    regular(workspace / specification['editable_preprocessor'], workspace)
