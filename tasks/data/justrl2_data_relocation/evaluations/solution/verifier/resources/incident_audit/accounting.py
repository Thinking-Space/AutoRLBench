"""Stdlib accounting primitives copied unchanged from the independent auditor.

Source: tasks/data/justrl2_data_relocation/evaluations/solution/verifier/resources/evidence_audit.py
Only strict JSON, per-step accounting, and checkpoint-schema helpers are retained.
No whole-run policy, hidden tests, model code, or repository-layout dependency.
"""
import hashlib
import json
import math
from pathlib import Path
import re


BASE_METRICS = {
    'data/read_attempts', 'data/abort_count', 'data/abort_rate', 'data/admitted_count',
    'data/admitted_hard_fraction', 'data/reward_sum', 'data/survivor_reward_mean',
    'data/admitted_reward_per_read_attempt',
}


TERMINAL_METRICS = {
    'data/terminal_count', 'data/requested_count', 'data/generation_abort_count',
    'data/scheduler_cancel_before_read', 'data/scored_count', 'data/scored_not_admitted',
    'data/truncated_count', 'data/all_scored_reward_sum', 'data/reward_per_read_attempt',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, context, minimum=0):
    _require(type(value) is int and value >= minimum, f'{context}: expected integer >= {minimum}')
    return value


def _number(value, context):
    _require(type(value) in (int, float) and math.isfinite(value), f'{context}: expected finite number')
    return value


def _reward(value, context='reward'):
    _number(value, context)
    _require(value in (0, 1), f'{context}: expected binary reward')
    return value


def _object(value, context):
    _require(isinstance(value, dict), f'{context}: expected object')
    return value


def _text(value, context):
    _require(isinstance(value, str) and bool(value.strip()), f'{context}: expected nonempty string')
    return value


def _digest(value, context):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), f'{context}: invalid SHA256')
    return value


def _hash(path):
    try:
        with path.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            return digest.hexdigest()
    except OSError as exc:
        raise ValueError(f'Missing/unreadable evidence {path}: {exc}') from exc


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'Duplicate JSON key {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Nonfinite JSON constant {value}')


def _decode(raw, context):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant,
                          parse_float=lambda value: _number(float(value), context))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid JSON in {context}: {exc}') from exc


def _load(path):
    try:
        return _object(_decode(path.read_text(), str(path)), str(path))
    except OSError as exc:
        raise ValueError(f'Missing/unreadable evidence {path}: {exc}') from exc


def _lines(path):
    try:
        with path.open() as stream:
            rows = [_object(_decode(line, f'{path}:{i}'), f'{path}:{i}')
                    for i, line in enumerate(stream, 1) if line.strip()]
    except OSError as exc:
        raise ValueError(f'Missing/unreadable evidence {path}: {exc}') from exc
    _require(rows, f'Empty evidence {path}')
    return rows


def _terminal_totals(events, reads, admitted, claimed):
    outcomes = {}
    positions = {(row['kind'], row['index']): i for i, row in enumerate(events)
                 if row['kind'] in ('read', 'admitted')}
    scored, generation_aborts, cancellations, truncated = [], 0, 0, 0
    for position, row in enumerate(events):
        if row['kind'] != 'terminal':
            continue
        index = _integer(row.get('index'), 'terminal index')
        _require(index not in outcomes, f'Duplicate terminal index {index}')
        outcomes[index] = row
        _text(row.get('id'), 'terminal identity')
        _require(row.get('difficulty') in ('easy', 'hard'), 'terminal difficulty is invalid')
        _integer(row.get('group_index'), 'terminal group_index')
        _integer(row.get('response_length'), 'terminal response_length')
        _require(isinstance(row.get('response'), str) and 'label' in row and 'reward' in row, 'terminal raw output fields are missing')
        read = reads.get(index)
        if read is not None:
            _require(all(row.get(k) == read.get(k) for k in ('id', 'group_index', 'difficulty')), 'terminal/read identity mismatch')
            _require(positions[('read', index)] < position, 'terminal precedes read')
        status = row.get('status')
        if status in ('completed', 'truncated'):
            _require(read is not None and read['outcome'] == 'ok', 'Scored terminal requires successful read')
            _reward(row['reward'], 'terminal reward')
            _integer(row['response_length'], 'scored terminal response_length', 1)
            scored.append(row)
            truncated += status == 'truncated'
        else:
            _require(status in ('aborted', 'pending') and row['reward'] is None and index not in admitted,
                     'Invalid terminal status/reward or aborted sample admitted')
            if read is None:
                cancellations += 1
            elif read['outcome'] == 'ok':
                generation_aborts += 1
            else:
                _require(status == 'aborted', 'Missing-file terminal must be aborted')
        if index in admitted:
            _require(positions[('admitted', index)] > position, 'Admission precedes terminal grading')
            _require(all(row.get(k) == admitted[index].get(k) for k in
                         ('id', 'group_index', 'difficulty', 'status', 'reward', 'response_length', 'response', 'label')),
                     'terminal/admitted identity or raw result mismatch')
    if not outcomes:
        return {}
    _require(reads.keys() <= outcomes.keys() and admitted.keys() <= outcomes.keys(), 'Read/admission missing terminal outcome')
    _require(TERMINAL_METRICS <= claimed.keys(), 'Missing terminal accounting metrics')
    requested = _integer(claimed['data/requested_count'], 'terminal requested_count', 1)
    _require(len(outcomes) == requested, 'Missing terminal outcomes for requested samples')
    reward_sum = sum(row['reward'] for row in scored)
    return {'data/terminal_count': len(outcomes), 'data/requested_count': len(outcomes),
            'data/generation_abort_count': generation_aborts, 'data/scheduler_cancel_before_read': cancellations,
            'data/scored_count': len(scored), 'data/scored_not_admitted': sum(row['index'] not in admitted for row in scored),
            'data/truncated_count': truncated, 'data/all_scored_reward_sum': reward_sum,
            'data/reward_per_read_attempt': reward_sum / len(reads)}


def audit_step(events, step):
    """Recompute admission/read denominators; successful surplus reads are valid.

    The small public step interface requires the terminal endpoint. Whole-run
    auditing additionally requires the producer's rollout_begin for every step.
    """
    _integer(step, 'step')
    rows = [e for e in events if _object(e, 'event').get('step') == step]
    endpoints = [e for e in rows if e.get('kind') == 'rollout_end']
    _require(len(endpoints) == 1, f'step {step}: expected one rollout_end endpoint')
    _require(rows[-1] is endpoints[0], f'step {step}: events after rollout_end')
    reads, admitted, seen_ids = {}, {}, {}
    for event in rows:
        kind = event.get('kind')
        if kind not in ('read', 'admitted'):
            _require(kind in ('rollout_begin', 'rollout_end', 'relocation', 'terminal'), f'Unknown event kind {kind}')
            continue
        index = _integer(event.get('index'), f'step {step} sample index')
        identifier = _text(event.get('id'), 'sample identity')
        difficulty = event.get('difficulty')
        _require(difficulty in ('easy', 'hard'), 'Unknown sample identity/difficulty')
        _require(identifier not in seen_ids or seen_ids[identifier] == difficulty, 'Sample identity/difficulty mismatch')
        seen_ids[identifier] = difficulty
        if kind == 'read':
            _require(index not in reads, f'Duplicate read index {index}')
            _require(event.get('outcome') in ('ok', 'missing'), 'Unknown file-read outcome')
            if event['outcome'] == 'missing':
                _require(type(event.get('errno')) is int and event['errno'] == 2, 'Missing read must have errno 2')
            reads[index] = event
        else:
            _require(index not in admitted, f'Duplicate admitted index {index}')
            read = reads.get(index)
            _require(read is not None and read['outcome'] == 'ok', f'Admission {index} requires an earlier successful read')
            _require(all(read.get(k) == event.get(k) for k in ('id', 'difficulty', 'group_index')),
                     f'Admission {index} identity/difficulty mismatch')
            _require(event.get('status') in ('completed', 'truncated'), 'Invalid admission status')
            _reward(event.get('reward'))
            admitted[index] = event
    _require(reads and admitted, f'step {step}: no read/admitted samples')
    missing = sum(r['outcome'] == 'missing' for r in reads.values())
    reward_sum = sum(r['reward'] for r in admitted.values())
    hard_count = sum(r['difficulty'] == 'hard' for r in admitted.values())
    computed = {
        'data/read_attempts': len(reads), 'data/abort_count': missing,
        'data/abort_rate': missing / len(reads), 'data/admitted_count': len(admitted),
        'data/read_ok_not_admitted': len(reads) - missing - len(admitted),
        'data/admitted_hard_fraction': hard_count / len(admitted), 'data/reward_sum': reward_sum,
        'data/survivor_reward_mean': reward_sum / len(admitted),
        'data/admitted_reward_per_read_attempt': reward_sum / len(reads),
    }
    group_reward_sums = {}
    for difficulty in ('easy', 'hard'):
        accepted = [r for r in admitted.values() if r['difficulty'] == difficulty]
        attempted = [r for r in reads.values() if r['difficulty'] == difficulty]
        group_reward_sums[difficulty] = sum(r['reward'] for r in accepted)
        computed[f'data/admitted_{difficulty}'] = len(accepted)
        computed[f'data/attempted_{difficulty}'] = len(attempted)
        computed[f'data/abort_{difficulty}'] = sum(r['outcome'] == 'missing' for r in attempted)
        if accepted:
            computed[f'data/reward_{difficulty}_mean'] = group_reward_sums[difficulty] / len(accepted)
    claimed = _object(endpoints[0].get('metrics'), 'rollout_end metrics')
    terminal_totals = _terminal_totals(rows, reads, admitted, claimed)
    computed.update(terminal_totals)
    _require(BASE_METRICS <= claimed.keys(), f'step {step}: missing required metric')
    for key, value in claimed.items():
        _require(key in computed, f'step {step}: unsupported metric {key}')
        _number(value, f'metric {key}')
        _require(math.isclose(value, computed[key], rel_tol=1e-9, abs_tol=1e-12),
                 f'step {step}: metric {key}={value}, recomputed {computed[key]}')
    return {'step': step, 'abort_rate': missing / len(reads), 'reward_mean': reward_sum / len(admitted),
            'hard_fraction': hard_count / len(admitted), 'read_attempts': len(reads), 'abort_count': missing,
            'admitted_count': len(admitted), 'hard_count': hard_count, 'reward_sum': reward_sum,
            'easy_reward_sum': group_reward_sums['easy'], 'hard_reward_sum': group_reward_sums['hard'],
            'terminal_accounting': bool(terminal_totals), 'metrics': computed}


def _inventory(value, context):
    value = _object(value, context)
    _require(value, f'{context}: empty file inventory')
    for name, record in value.items():
        _text(name, f'{context} filename')
        _require(not Path(name).is_absolute() and '..' not in Path(name).parts, f'{context}: invalid relative filename')
        _object(record, f'{context} {name}')
        _integer(record.get('size'), f'{context} {name} size', 1)
        _digest(record.get('sha256'), f'{context} {name} SHA256')
    return value


def _checkpoint(record, iteration, require_optimizer):
    _object(record, 'checkpoint receipt')
    selected = 'release' if iteration == 'release' else f'iter_{int(iteration):07d}'
    _require(record.get('iteration') == iteration and record.get('selected') == selected,
             'checkpoint tracker/selected iteration does not match current run budget')
    keys = record.get('state_keys')
    _require(isinstance(keys, list) and keys and all(isinstance(key, str) for key in keys), 'checkpoint has no state keys')
    _require(any(key.startswith(('model.', 'model0.', 'embedding.', 'decoder.', 'output_layer.')) for key in keys),
             'checkpoint metadata has no model state')
    if require_optimizer:
        _require(any('optim' in key.lower() for key in keys), 'checkpoint metadata has no optimizer state')
    files = _inventory(record.get('referenced_files'), 'checkpoint referenced files')
    _require('.metadata' in files and any(name.endswith('.distcp') for name in files), 'checkpoint requires metadata and referenced payload hashes')
    _require(type(record.get('restore_tested')) is bool, 'checkpoint restore_tested flag is missing')
    return files
