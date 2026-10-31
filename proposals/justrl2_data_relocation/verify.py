"""Independent, stdlib-only audit of the data-relocation evidence contract.

Never imports the producer's summarizer or executes native training/eval code.
A single complete run is a valid pilot, not a three-arm causal completion.
This checks supplied evidence for consistency; it does not authenticate its author.
"""
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tokenize


ARMS = ('healthy', 'faulty', 'repaired')
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


def _audit_inputs(home):
    inputs = _load(home / 'INPUTS.json')
    _require(inputs.get('synthetic_fault_injection') is True, 'INPUTS must label the isolated fault injection')
    _require(inputs.get('arm') in ARMS, 'INPUTS arm is invalid')
    _text(inputs.get('job_id'), 'INPUTS job_id')
    steps = _integer(inputs.get('steps'), 'INPUTS steps', 2)
    at = _integer(inputs.get('relocate_at'), 'INPUTS relocate_at', 1)
    _require(steps == 2 * at, 'INPUTS relocation must be at the midpoint of an even step budget')
    for key in ('model_revision', 'source_revision'):
        _text(inputs.get(key), f'INPUTS {key}')
    for key in ('model_files', 'plugin_files'):
        inventory = _object(inputs.get(key), f'INPUTS {key}')
        _require(inventory, f'INPUTS {key} cannot be empty')
        for name, digest in inventory.items():
            _text(name, f'INPUTS {key} filename')
            _digest(digest, f'INPUTS {key}/{name}')
    for key in ('case_sha256', 'manifest_sha256', 'eval_sha256'):
        _digest(inputs.get(key), f'INPUTS {key}')
    return inputs


def _audit_config(config, inputs):
    expected = {
        'debug_train_only': False, 'debug_rollout_only': False, 'debug_disable_optimizer': False,
        'debug_skip_weight_update': False, 'load_debug_rollout_data': None,
        'partial_rollout': False, 'use_critic': True, 'rollout_global_dataset': True,
        'no_load_optim': True, 'no_load_rng': True, 'finetune': True,
        'start_rollout_id': 0, 'update_weights_interval': 1, 'num_critic_only_steps': 0,
        'num_rollout': inputs['steps'], 'eval_interval': inputs['relocate_at'], 'skip_eval_before_train': False,
        'train_backend': 'megatron', 'advantage_estimator': 'ppo', 'rm_type': 'math',
        'data_source_path': 'relocation_rollout.BackedMathSource',
        'rollout_function_path': 'relocation_rollout.generate_rollout',
        'eval_function_path': 'relocation_rollout.generate_rollout',
        'custom_generate_function_path': 'relocation_rollout.generate',
        'dynamic_sampling_filter_path': 'miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted',
        'actor_num_nodes': 1, 'actor_num_gpus_per_node': 1, 'critic_num_nodes': 1,
        'critic_num_gpus_per_node': 1, 'rollout_num_gpus': 1,
        'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1,
    }
    for key, value in expected.items():
        _require(key in config and type(config[key]) is type(value) and config[key] == value,
                 f'effective config {key}: expected {value!r}')
    _integer(config['eval_interval'], 'config eval_interval', 1)
    for key in ('rollout_batch_size', 'n_samples_per_prompt', 'global_batch_size', 'micro_batch_size',
                'rollout_max_response_len', 'eval_max_response_len', 'n_samples_per_eval_prompt'):
        _integer(config.get(key), f'config {key}', 1)
    _require(config.get('over_sampling_batch_size') == config['rollout_batch_size'], 'config oversampling budget mismatch')
    _require(config['global_batch_size'] == config['rollout_batch_size'] * config['n_samples_per_prompt'],
             'config global batch must match the native single-update recipe')
    for key in ('seed', 'rollout_seed'):
        _integer(config.get(key), f'config {key}')
    for key in ('lr', 'critic_lr'):
        _require(_number(config.get(key), f'config {key}') > 0, f'config {key} must be positive')
    for key in ('hf_checkpoint', 'ref_load', 'load', 'critic_load', 'save', 'critic_save'):
        _text(config.get(key), f'config {key}')
    _require(config['load'] == config['critic_load'] == config['ref_load'], 'config must start fresh from reference weights')
    env = _object(config.get('train_env_vars'), 'config train_env_vars')
    for key, value in (('RELOCATION_ARM', inputs['arm']), ('RELOCATION_STEPS', str(inputs['steps'])),
                       ('RELOCATION_AT', str(inputs['relocate_at']))):
        _require(env.get(key) == value, f'config environment {key} disagrees with INPUTS')
    run_path = str(Path(config['save']).parent)
    _require(Path(run_path).name == 'run' and config['save'] == str(Path(run_path) / 'actor') and
             config['critic_save'] == str(Path(run_path) / 'critic') and env.get('RELOCATION_OUTPUT') == run_path,
             'config output paths disagree')
    data_path = _text(env.get('RELOCATION_DATA'), 'config RELOCATION_DATA')
    _require(data_path == str(Path(run_path).parent / 'dataset'), 'config generation must use this run home dataset')
    _require(config.get('eval_prompt_data') == ['fixed', str(Path(data_path) / 'eval.jsonl')],
             'config must use the fixed eval dataset')


def _audit_dataset(home, inputs):
    data = home / 'dataset'
    provenance = home / 'provenance'
    exported = provenance.exists()
    _require(exported or 'native_checkpoint_files' not in inputs, 'Missing full provenance directory')
    evidence = provenance if exported else data
    manifest_path = evidence / ('final_manifest.json' if exported else 'manifest.json')
    if data.exists():
        if inputs['arm'] == 'healthy':
            _require(not (data / 'archive/hard').exists(), 'healthy storage contains a relocation')
        else:
            _require(not (data / 'live/hard').exists(), 'relocation did not remove the live hard directory')
    for filename, key in (('CASE.json', 'case_sha256'), ('eval.jsonl', 'eval_sha256')):
        _require(_hash(evidence / filename) == inputs[key], f'INPUTS {filename} hash mismatch')
        if exported and data.exists():
            _require(_hash(data / filename) == inputs[key], f'Current dataset {filename} disagrees with provenance')
    case = _load(evidence / 'CASE.json')
    _require(case.get('synthetic_fault_injection') is True and case.get('schema_version') == 1, 'Invalid CASE marker')
    _digest(_object(case.get('source'), 'CASE source').get('sha256'), 'CASE source SHA256')
    provenance = case.get('rows')
    _require(isinstance(provenance, list) and provenance, 'Missing CASE source IDs')
    by_id = {}
    for row in provenance:
        _object(row, 'CASE row')
        identifier = _text(row.get('id'), 'CASE ID')
        _require(identifier not in by_id, 'Duplicate CASE source ID')
        _require(row.get('difficulty') in ('easy', 'hard') and row.get('split') in ('train', 'eval'), 'Invalid CASE identity')
        _integer(row.get('source_line'), 'CASE source_line', 1)
        for key in ('prompt_sha256', 'label_sha256'):
            _digest(row.get(key), f'CASE {key}')
        by_id[identifier] = row
    manifest = _load(manifest_path)
    if exported and data.exists():
        _require(_hash(data / 'manifest.json') == _hash(manifest_path), 'Current manifest disagrees with provenance')
    rows = manifest.get('rows')
    _require(manifest.get('schema_version') == 1 and isinstance(rows, list) and rows, 'Invalid manifest')
    initial_rows, payloads = [], {}
    for row in rows:
        _object(row, 'manifest row')
        identifier = _text(row.get('id'), 'manifest ID')
        _require(re.fullmatch(r'[A-Za-z0-9_-]+', identifier), 'Invalid manifest ID')
        _require(identifier not in payloads, 'Duplicate manifest source ID')
        source = by_id.get(identifier)
        _require(source is not None and all(source.get(k) == row.get(k) for k in ('difficulty', 'split', 'source_line'))
                 and row.get('split') == 'train', 'Manifest source identity mismatch')
        difficulty = row['difficulty']
        original = f'live/{difficulty}/{identifier}.json'
        archived = f'archive/hard/{identifier}.json'
        expected_path = archived if inputs['arm'] == 'repaired' and difficulty == 'hard' else original
        _require(row.get('path') == expected_path, 'Manifest mapping does not match relocation arm')
        actual_path = archived if inputs['arm'] != 'healthy' and difficulty == 'hard' else original
        _digest(row.get('sha256'), 'backing SHA256')
        if data.exists():
            _require(_hash(data / actual_path) == row['sha256'], f'backing hash mismatch for {identifier}')
            payload = _load(data / actual_path)
            metadata = _object(payload.get('metadata'), 'backing metadata')
            _require(payload.get('id') == identifier and all(metadata.get(k) == row[k] for k in ('difficulty', 'source_line')),
                     'Backing source identity mismatch')
            _require(isinstance(payload.get('prompt'), list) and payload['prompt'] and 'label' in payload, 'Missing backing prompt/label')
            _require(_content_hash(payload['label']) == source['label_sha256'] and
                     _content_hash(payload['prompt']) == source['prompt_sha256'], 'Backing prompt/label differs from source provenance')
        payloads[identifier] = {'label_sha256': source['label_sha256']}
        initial_rows.append({**row, 'path': original})
    _require(set(payloads) == {r['id'] for r in provenance if r['split'] == 'train'}, 'Manifest deleted source IDs')
    _require({r['difficulty'] for r in rows} == {'easy', 'hard'}, 'Manifest must retain both difficulties')
    initial = {**manifest, 'rows': initial_rows}
    # storage.py writes this exact sorted, indented representation before repair.
    initial_bytes = (json.dumps(initial, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(hashlib.sha256(initial_bytes).hexdigest() == inputs['manifest_sha256'], 'INPUTS initial manifest hash mismatch')
    if exported:
        _require(_hash(evidence / 'initial_manifest.json') == inputs['manifest_sha256'] and
                 _load(evidence / 'initial_manifest.json') == initial, 'Initial manifest provenance does not match INPUTS/final mapping')
    evaluation = _lines(evidence / 'eval.jsonl')
    eval_ids = []
    for row in evaluation:
        source = by_id.get(row.get('id'))
        metadata = _object(row.get('metadata'), 'eval dataset metadata')
        _require(source is not None and source['split'] == 'eval' and
                 all(source[k] == metadata.get(k) for k in ('difficulty', 'source_line')), 'eval dataset source identity mismatch')
        _require(row['id'] not in eval_ids and row['id'] not in payloads, 'Duplicate/overlapping eval source IDs')
        _require('label' in row and isinstance(row.get('prompt'), list) and row['prompt'], 'Missing eval prompt/label')
        _require(_content_hash(row['label']) == source['label_sha256'] and _content_hash(row['prompt']) == source['prompt_sha256'],
                 'eval prompt/label differs from source provenance')
        eval_ids.append(row['id'])
    _require(set(eval_ids) == {r['id'] for r in provenance if r['split'] == 'eval'}, 'Missing eval source IDs')
    _require({r['metadata']['difficulty'] for r in evaluation} == {'easy', 'hard'}, 'eval must retain both difficulties')
    return initial_rows, payloads, evaluation, _hash(manifest_path)


def _audit_relocations(events, inputs, manifest, after_hash):
    relocations = [e for e in events if e['kind'] == 'relocation']
    if inputs['arm'] == 'healthy':
        _require(not relocations, 'healthy arm must have no relocation')
        return
    at, steps = inputs['relocate_at'], inputs['steps']
    _require([e['step'] for e in relocations] == list(range(at, steps)),
             'relocation must first occur at midpoint and retain receipts each post step')
    if manifest is None:
        # Old exports omitted the dataset. Check receipt consistency only;
        # audit_run explicitly reports source_identity_verified=False.
        first = _object(relocations[0].get('receipt'), 'legacy relocation receipt')
        files = first.get('files')
        _require(isinstance(files, list) and files, 'legacy relocation receipt has no files')
        identifiers = set()
        manifest = []
        for row in files:
            _object(row, 'legacy relocation file')
            identifier = _text(row.get('id'), 'legacy relocation ID')
            _require(re.fullmatch(r'[A-Za-z0-9_-]+', identifier) and identifier not in identifiers, 'legacy relocation duplicate/invalid ID')
            identifiers.add(identifier)
            _digest(row.get('sha256'), 'legacy relocation SHA256')
            manifest.append({'id': identifier, 'difficulty': 'hard', 'sha256': row['sha256'],
                             'path': f'live/hard/{identifier}.json'})
        after_hash = _digest(first.get('manifest_after_sha256'), 'legacy relocation manifest SHA256')
    expected = {'action': 'relocate', 'repair': inputs['arm'] == 'repaired',
                'source': 'live/hard', 'destination': 'archive/hard',
                'manifest_before_sha256': inputs['manifest_sha256'], 'manifest_after_sha256': after_hash,
                'files': [{'id': r['id'], 'sha256': r['sha256'], 'source': r['path'],
                           'destination': f"archive/hard/{r['id']}.json"} for r in manifest if r['difficulty'] == 'hard']}
    expected['rows_moved'] = len(expected['files'])
    for event in relocations:
        _require(event.get('receipt') == expected, 'relocation receipt does not match original IDs/content/manifest hashes')
    _require((after_hash != inputs['manifest_sha256']) == (inputs['arm'] == 'repaired'), 'relocation manifest repair mismatch')


def _audit_events(events, inputs, config, manifest, payloads, after_hash):
    steps, at, arm = inputs['steps'], inputs['relocate_at'], inputs['arm']
    for event in events:
        _integer(event.get('step'), 'event step')
        _require(event['step'] < steps and event.get('arm') == arm, 'Event step/arm outside INPUTS run')
        _require(event.get('kind') in ('rollout_begin', 'read', 'admitted', 'rollout_end', 'relocation', 'terminal'), 'Unknown/missing event kind')
    _audit_relocations(events, inputs, manifest, after_hash)
    _require([e['step'] for e in events] == sorted(e['step'] for e in events), 'Events mix step order')
    seen_indices, previous_max_group, next_cursor, observed_ids = set(), -1, 0, {}
    has_terminals = any(e['kind'] == 'terminal' for e in events)
    results = []
    n = config['n_samples_per_prompt']
    for step in range(steps):
        rows = [e for e in events if e['step'] == step]
        begins = [e for e in rows if e['kind'] == 'rollout_begin']
        _require(len(begins) == 1, f'step {step}: missing/duplicate rollout_begin endpoint')
        begin_position = 1 if arm != 'healthy' and step >= at else 0
        _require(rows[begin_position] is begins[0], f'step {step}: misplaced rollout_begin/relocation')
        cursor = _integer(begins[0].get('cursor'), 'rollout_begin cursor')
        _require(cursor > previous_max_group and (step != 0 or cursor == 0), 'Run is not fresh or sampling cursor rewound')
        result = audit_step(rows, step)
        _require(result['terminal_accounting'] == has_terminals, 'Missing terminal accounting in one or more steps')
        if has_terminals:
            _require(cursor == next_cursor, 'terminal requested sample cursor has gaps between steps')
            count = result['metrics']['data/requested_count']
            _require(count % n == 0 and count <= 16 * config['rollout_batch_size'] * n, 'terminal requested count violates sampling budget')
            expected_indices = set(range(cursor * n, cursor * n + count))
            terminals = [e for e in rows if e['kind'] == 'terminal']
            _require({e['index'] for e in terminals} == expected_indices, 'terminal accounting does not cover all intended requested indices')
            for event in terminals:
                source = manifest[event['index'] // n % len(manifest)]
                _require(event['group_index'] == event['index'] // n and
                         all(event[k] == source[k] for k in ('id', 'difficulty')), 'terminal source identity/intended sampling mismatch')
                if event['status'] in ('completed', 'truncated'):
                    _require(event['response_length'] <= config['rollout_max_response_len'], 'terminal response exceeds generation budget')
                    _require(_content_hash(event['label']) == payloads[event['id']]['label_sha256'], 'terminal source label mismatch')
            next_cursor = cursor + count // n
        _require(result['admitted_count'] == config['global_batch_size'], 'Admitted batch/config budget mismatch')
        groups = {}
        for event in rows:
            if event['kind'] == 'read':
                index = event['index']
                _require(index not in seen_indices, f'Duplicate read index across steps: {index}')
                seen_indices.add(index)
                group = _integer(event.get('group_index'), 'read group_index')
                _require(group >= cursor and index // n == group, 'Source sampling group/index mismatch')
                if manifest is not None:
                    source = manifest[group % len(manifest)]
                    _require(all(event.get(k) == source[k] for k in ('id', 'difficulty')), 'Read source identity/intended sampling mismatch')
                else:
                    identifier = event['id']
                    _require(re.fullmatch(r'[A-Za-z0-9_-]+', identifier), 'Invalid legacy source ID')
                    source = {'id': identifier, 'difficulty': event['difficulty'],
                              'path': f"live/{event['difficulty']}/{identifier}.json", 'sha256': event.get('sha256')}
                    if event['outcome'] == 'ok':
                        _digest(source['sha256'], 'legacy read SHA256')
                    previous = observed_ids.get(identifier)
                    _require(previous is None or previous['difficulty'] == source['difficulty'], 'Legacy source difficulty changed')
                    if previous and previous['sha256'] and source['sha256']:
                        _require(previous['sha256'] == source['sha256'], 'Legacy source content hash changed')
                    observed_ids[identifier] = source
                path = source['path']
                if arm == 'repaired' and step >= at and source['difficulty'] == 'hard':
                    path = f"archive/hard/{source['id']}.json"
                _require(event.get('path') == path, 'Read path disagrees with phase/manifest')
                missing = arm == 'faulty' and step >= at and source['difficulty'] == 'hard'
                _require(event['outcome'] == ('missing' if missing else 'ok'), 'Read outcome disagrees with actual relocation/phase')
                if event['outcome'] == 'ok':
                    _require(event.get('sha256') == source['sha256'], 'Read SHA256 disagrees with original source')
                else:
                    _require(arm == 'faulty' and step >= at and source['difficulty'] == 'hard', 'Unexpected file abort outside faulty post phase')
                previous_max_group = max(previous_max_group, group)
            elif event['kind'] == 'admitted':
                _text(event.get('response'), 'admitted raw response')
                length = _integer(event.get('response_length'), 'admitted response_length', 1)
                _require(length <= config['rollout_max_response_len'], 'Response exceeds config generation budget')
                if payloads is not None:
                    _require(_content_hash(event.get('label')) == payloads[event['id']]['label_sha256'], 'Admitted source label mismatch')
                else:
                    _require('label' in event, 'Missing legacy admitted label')
                groups.setdefault(event['group_index'], []).append(event['index'])
        _require(len(groups) == config['rollout_batch_size'] and all(
            sorted(indices) == list(range(group * n, (group + 1) * n)) for group, indices in groups.items()),
            'Admitted native groups are incomplete')
        results.append(result)
    return results


def _audit_optimizer(run, steps):
    records = []
    for path in sorted((run / 'optimizer_evidence').glob('*.jsonl')):
        records.extend(_lines(path))
    _require(records, 'Missing native optimizer evidence')
    seen, coverage = set(), {role: set() for role in ('actor', 'critic')}
    changed = {role: False for role in coverage}
    for row in records:
        role = row.get('role')
        _require(row.get('event') == 'optimizer_step' and role in coverage, 'Invalid native optimizer event/role')
        rollout = _integer(row.get('rollout_id'), 'optimizer rollout_id')
        step_id = _integer(row.get('step_id'), 'optimizer step_id')
        rank = _integer(row.get('rank'), 'optimizer rank')
        _require(rollout < steps and rank == 0, 'optimizer evidence outside configured steps/ranks')
        identity = (role, rank, rollout, step_id)
        _require(identity not in seen, 'Duplicate optimizer update')
        seen.add(identity)
        for key in ('update_successful', 'grad_norm_finite', 'sampled_finite'):
            _require(row.get(key) is True, f'optimizer {role}/{rollout}: {key} must be true')
        _require(_number(row.get('grad_norm'), 'optimizer grad_norm') >= 0, 'optimizer negative gradient norm')
        _require(type(row.get('sampled_changed')) is bool, 'optimizer sampled_changed must be boolean')
        count = _integer(row.get('sampled_tensors'), 'optimizer sampled_tensors', 1)
        elements = _integer(row.get('sampled_elements'), 'optimizer sampled_elements', 1)
        changed_count = _integer(row.get('changed_tensors'), 'optimizer changed_tensors')
        delta = _number(row.get('max_delta'), 'optimizer max_delta')
        _require(row.get('sampled_bytes') == elements * 4 and changed_count <= count and delta >= 0,
                 'optimizer sampled counts/delta mismatch')
        _require(row['sampled_changed'] == (changed_count > 0) == (delta > 0), 'optimizer sampled change evidence contradicts itself')
        coverage[role].add(rollout)
        changed[role] |= row['sampled_changed']
    for role in coverage:
        _require(coverage[role] == set(range(steps)), f'optimizer {role} is missing per-step evidence')
        _require(changed[role], f'optimizer {role} has no sampled_changed update')
    return {role: {'steps': sorted(coverage[role]), 'sampled_changed': changed[role]} for role in coverage}


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


def _audit_provenance(home, inputs, config, events, steps):
    for relative in ('RESULT.json', 'export/RESULT.json'):
        result_path = home / relative
        if result_path.exists():
            result = _load(result_path)
            _require(type(result.get('returncode')) is int and result['returncode'] == 0 and
                     result.get('job_id') == inputs['job_id'],
                     f'{relative}: completion/audit failed despite TRAINING_RETURNED')
    directory = home / 'provenance'
    if not directory.exists():
        return {'complete': False}
    source = _load(directory / 'source_receipt.json')
    _require(source.get('revision') == inputs['source_revision'], 'Source provenance revision differs from INPUTS')
    _digest(source.get('source_archive_sha256'), 'Source archive SHA256')
    inventories = []
    for key in ('original_files', 'instrumented_files', 'helpers'):
        inventory = _object(source.get(key), f'Source provenance {key}')
        _require(inventory, f'Source provenance {key} is empty')
        for digest in inventory.values():
            _digest(digest, f'Source provenance {key} SHA256')
        inventories.append(inventory)
    _require(inventories[0].keys() == inventories[1].keys(), 'Source inventory lost files during instrumentation')
    conversion = _load(directory / 'conversion_receipt.json')
    for key in ('model_revision', 'source_revision'):
        _require(conversion.get(key) == inputs[key], f'Conversion provenance {key} differs from INPUTS')
    for key in ('source_inventory_sha256', 'conversion_log_sha256'):
        _digest(conversion.get(key), f'Conversion provenance {key}')
    native = _checkpoint(conversion.get('native_checkpoint'), 'release', False)
    before = _inventory(inputs.get('native_checkpoint_files'), 'INPUTS native_checkpoint_files')
    after = _inventory(_load(directory / 'original_native_unchanged.json'), 'Original native checkpoint after training')
    _require(before == native == after, 'Original native checkpoint hashes changed or do not match conversion')
    converter = _object(conversion.get('converter_input'), 'Conversion input')
    files = _inventory(converter.get('files'), 'Conversion input files')
    _require({name: row['sha256'] for name, row in files.items()} == inputs['model_files'],
             'Conversion model file hashes disagree with INPUTS')
    tensors = _object(converter.get('tensor_check'), 'Conversion tensor check')
    hf_files = _object(conversion.get('hf_files'), 'Conversion public HF hashes')
    _digest(hf_files.get('model/pytorch_model.bin'), 'Original public HF weight SHA256')
    _digest(tensors.get('source_sha256'), 'Conversion original weight SHA256')
    for digest in hf_files.values():
        _digest(digest, 'Public HF SHA256')
    _require(converter.get('format') == 'losslessly_reencoded_safetensors' and tensors.get('all_tensors_exactly_equal') is True,
             'Conversion lacks exact tensor equivalence evidence')
    _require('model.safetensors' in files and tensors.get('source_sha256') == hf_files.get('model/pytorch_model.bin') and
             tensors.get('target_sha256') == files['model.safetensors']['sha256'] and
             tensors.get('target_size') == files['model.safetensors']['size'], 'Conversion tensor hashes/sizes are not bound to original weights')
    _integer(tensors.get('tensor_count'), 'Conversion tensor count', 1)
    checkpoints = _load(directory / 'output_checkpoints.json')
    _require(set(checkpoints) == {'actor', 'critic'}, 'Missing actor/critic output checkpoint provenance')
    rehashed = {}
    for role, record in checkpoints.items():
        files = _checkpoint(record, str(inputs['steps'] - 1), True)
        root = home / 'run' / role / record['selected']
        rehashed[role] = all((root / name).is_file() for name in files)
        for name, expected in files.items():
            path = root / name
            if path.exists():
                _require(path.is_file() and path.stat().st_size == expected['size'] and _hash(path) == expected['sha256'],
                         f'Output checkpoint {role}/{name} hash/size mismatch')
    relocation_path = directory / 'final_relocation.json'
    if inputs['arm'] == 'healthy':
        _require(not relocation_path.exists(), 'Healthy provenance contains a relocation')
    else:
        journal = _load(relocation_path)
        receipt = next(e['receipt'] for e in events if e['kind'] == 'relocation')
        _require(journal.get('schema_version') == 1 and 'pending' in journal and journal['pending'] is None and
                 journal.get('events') == [receipt], 'Final relocation provenance is incomplete or disagrees with events')
    last = inputs['steps'] - 1
    state = _load(home / 'run/actor/rollout' / f'data_state_{last}.json')
    cursor = _integer(state.get('cursor'), 'Final source cursor')
    _require(state.get('group_index') == cursor and state.get('sample_index') == cursor * config['n_samples_per_prompt'] and
             state.get('step') == last and state.get('arm') == inputs['arm'], 'Final source state/counts disagree with run')
    if all(step['terminal_accounting'] for step in steps):
        requested = sum(step['metrics']['data/requested_count'] for step in steps)
        _require(state['sample_index'] == requested, 'Final source checkpoint does not account for every requested terminal sample')
    return {'complete': True, 'source_receipt_sha256': _hash(directory / 'source_receipt.json'),
            'conversion_receipt_sha256': _hash(directory / 'conversion_receipt.json'),
            'checkpoint_payloads_rehashed': rehashed, 'final_source_sample_index': state['sample_index'],
            'checkpoint_iteration': last, 'restore_tested': {role: row['restore_tested'] for role, row in checkpoints.items()}}


def _repr_fields(raw, name):
    """Split native dataclass repr fields with Python's lexer, never eval().

    Strings (including commas/quotes in model outputs) remain opaque tokens.
    Nested dataclasses are left uninterpreted. Enum repr is handled separately.
    """
    _require(isinstance(raw, str) and raw.startswith(name + '(') and raw.endswith(')'), f'eval: expected native {name}(...) repr')
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(raw[len(name) + 1:-1]).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ValueError('eval: malformed native repr') from exc
    fields, segment, depth = {}, [], 0

    def consume():
        if not segment:
            return
        _require(len(segment) >= 3 and segment[0][0] == tokenize.NAME and segment[1][1] == '=', 'eval: malformed repr field')
        key = segment[0][1]
        _require(key not in fields, 'eval: duplicate repr field')
        fields[key] = tokenize.untokenize(segment[2:]).strip()
        segment.clear()

    for token in tokens:
        if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL):
            continue
        if token.type == tokenize.OP:
            if token.string == ',' and depth == 0:
                consume()
                continue
            if token.string in ('(', '[', '{'):
                depth += 1
            elif token.string in (')', ']', '}'):
                depth -= 1
                _require(depth >= 0, 'eval: malformed repr nesting')
        segment.append((token.type, token.string))
    _require(depth == 0, 'eval: malformed repr nesting')
    consume()
    return fields


def _literal(fields, key):
    _require(key in fields, f'eval: missing native sample/config field {key}')
    try:
        return ast.literal_eval(fields[key])
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f'eval: unsupported literal {key}') from exc


def _eval_sample(raw, structured):
    if structured:
        return _object(raw, 'eval structured sample')
    fields = _repr_fields(raw, 'Sample')
    sample = {key: _literal(fields, key) for key in
              ('index', 'reward', 'response', 'response_length', 'prompt', 'label', 'metadata')}
    match = re.fullmatch(r"<\s*Status\s*\.\s*(COMPLETED|TRUNCATED)\s*:\s*['\"](completed|truncated)['\"]\s*>",
                         fields.get('status', ''))
    _require(match is not None and match[1].lower() == match[2], 'eval: invalid native sample status')
    sample['status'] = match[2]
    return sample


def _audit_eval(run, config, evaluation, steps):
    interval = config['eval_interval']
    paths = list(run.glob('eval_*.json'))
    structured = any(p.name.startswith(('eval_before_', 'eval_after_')) for p in paths)
    periodic = list(range(interval - 1, steps, interval))
    if structured:
        _require(evaluation is not None, 'Structured eval requires source provenance')
        snapshots = [('eval_before_train.json', None),
                     *[(f'eval_after_step_{step:07d}.json', step) for step in periodic]]
    else:
        _require(interval >= 2, 'eval_interval=1 overwrites initial eval_0; phase-explicit schema required')
        snapshots = [('eval_0.json', None), *[(f'eval_{step}.json', step) for step in periodic]]
    _require({p.name for p in paths} == {name for name, _ in snapshots}, 'eval: missing/extra raw phase endpoints')
    datasets = config.get('eval_datasets')
    _require(isinstance(datasets, list) and len(datasets) == 1, 'eval: expected one native fixed dataset config')
    dataset = _repr_fields(datasets[0], 'EvalDatasetConfig')
    _require(_literal(dataset, 'name') == 'fixed' and _literal(dataset, 'path') == config['eval_prompt_data'][1],
             'eval: native dataset config/path mismatch')
    count = _literal(dataset, 'n_samples_per_eval_prompt')
    _require(type(count) is int and count == config['n_samples_per_eval_prompt'], 'eval: native sample budget mismatch')
    summaries, reference = [], None
    for filename, step in snapshots:
        output = _load(run / filename)
        _require(set(output) == {'fixed'}, 'eval: expected raw fixed dataset output')
        fixed = _object(output['fixed'], 'eval fixed')
        rewards, truncated, samples = (fixed.get(k) for k in ('rewards', 'truncated', 'samples'))
        expected_count = len(evaluation) * count if evaluation is not None else len(samples) if isinstance(samples, list) else 0
        _require(expected_count > 0 and all(isinstance(v, list) and len(v) == expected_count for v in (rewards, truncated, samples)),
                 'eval: raw rewards/truncated/samples missing or wrong length')
        identities = []
        for index, raw in enumerate(samples):
            sample = _eval_sample(raw, structured)
            identity = sample.get('index')
            _require(type(identity) is int and identity == index, 'eval: duplicate/out-of-order sample index')
            reward = _reward(sample.get('reward'), 'eval sample reward')
            _require(_reward(rewards[index], 'eval reward') == reward, 'eval: raw sample/rewards mismatch')
            status = sample.get('status')
            _require(status in ('completed', 'truncated'), 'eval: invalid native sample status')
            _require(type(truncated[index]) is bool and truncated[index] == (status == 'truncated'), 'eval: truncated flag mismatch')
            _text(sample.get('response'), 'eval raw response')
            length = _integer(sample.get('response_length'), 'eval response_length', 1)
            _require(length <= config['eval_max_response_len'], 'eval: response exceeds budget')
            prompt, label = sample.get('prompt'), sample.get('label')
            _require(isinstance(prompt, (str, list)) and prompt, 'eval: missing raw prompt')
            metadata = _object(sample.get('metadata'), 'eval sample metadata')
            _integer(metadata.get('source_line'), 'eval source_line', 1)
            _require(metadata.get('difficulty') in ('easy', 'hard'), 'eval: invalid difficulty')
            source = evaluation[index // count] if evaluation is not None else None
            if source is not None:
                _require(label == source['label'] and all(metadata.get(k) == source['metadata'][k] for k in ('source_line', 'difficulty')),
                         'eval: source identity/label/difficulty mismatch')
            identities.append({'index': index, 'id': source['id'] if source else None, 'prompt': prompt, 'label': label,
                               'difficulty': metadata['difficulty'], 'source_line': metadata['source_line']})
        _require(reference is None or identities == reference, 'eval: fixed sample identities changed across phases')
        reference = identities
        summaries.append({'rollout_id': step, 'phase': 'initial' if step is None else 'pre' if step < interval else 'post',
                          'count': len(rewards), 'reward_mean': sum(rewards) / len(rewards)})
    return summaries, reference, 'structured' if structured else 'legacy'


def _phase(steps):
    attempts = sum(s['read_attempts'] for s in steps)
    admitted = sum(s['admitted_count'] for s in steps)
    aborted = sum(s['abort_count'] for s in steps)
    hard = sum(s['hard_count'] for s in steps)
    reward = sum(s['reward_sum'] for s in steps)
    result = {'read_attempts': attempts, 'abort_count': aborted, 'abort_rate': aborted / attempts,
            'admitted_count': admitted, 'hard_count': hard, 'hard_fraction': hard / admitted,
            'reward_sum': reward, 'reward_mean': reward / admitted,
            'admitted_reward_per_read_attempt': reward / attempts}
    for difficulty, count in (('easy', admitted - hard), ('hard', hard)):
        total = sum(s[f'{difficulty}_reward_sum'] for s in steps)
        result[f'{difficulty}_count'] = count
        result[f'{difficulty}_reward_sum'] = total
        result[f'{difficulty}_reward_mean'] = total / count if count else None
    if all(s['terminal_accounting'] for s in steps):
        for name in TERMINAL_METRICS - {'data/reward_per_read_attempt'}:
            result[name.removeprefix('data/')] = sum(s['metrics'][name] for s in steps)
        result['reward_per_read_attempt'] = result['all_scored_reward_sum'] / attempts
    return result


def audit_run(home):
    """Audit a full local run home (including dataset); raise on missing evidence."""
    home = Path(home)
    inputs = _audit_inputs(home)
    run = home / 'run'
    returned = _load(run / 'TRAINING_RETURNED.json')
    _require(returned.get('training_returned') is True and returned.get('arm') == inputs['arm'] and
             type(returned.get('physical_gpus')) is int and returned['physical_gpus'] == 3, 'Invalid TRAINING_RETURNED completion')
    _require(_number(returned.get('elapsed_seconds'), 'TRAINING_RETURNED elapsed_seconds') > 0, 'Invalid completion elapsed time')
    packages = _object(returned.get('packages'), 'TRAINING_RETURNED packages')
    for name in ('torch', 'ray', 'sglang', 'transformers'):
        _text(packages.get(name), f'TRAINING_RETURNED package {name}')
    config = _load(run / 'effective_config.json')
    _audit_config(config, inputs)
    events = _lines(run / 'events.jsonl')
    source_verified = (home / 'dataset').exists() or (home / 'provenance').exists()
    if source_verified:
        manifest, payloads, evaluation, after_hash = _audit_dataset(home, inputs)
    else:
        _require('native_checkpoint_files' not in inputs and not any(e.get('kind') == 'terminal' for e in events) and
                 not any(run.glob('eval_before_*.json')) and not any(run.glob('eval_after_*.json')),
                 'Full evidence export is missing source/data provenance')
        manifest = payloads = evaluation = after_hash = None
    steps = _audit_events(events, inputs, config, manifest, payloads, after_hash)
    optimizer = _audit_optimizer(run, inputs['steps'])
    eval_results, eval_identity, eval_format = _audit_eval(run, config, evaluation, inputs['steps'])
    provenance = _audit_provenance(home, inputs, config, events, steps)
    at = inputs['relocate_at']
    return {'valid': True, 'status': 'pilot_valid', 'causal_complete': False,
            'synthetic_fault_injection': True, 'home': str(home), 'arm': inputs['arm'],
            'inputs': inputs, 'config': config, 'packages': packages,
            'source_ids': [r['id'] for r in manifest] if manifest is not None else None,
            'source_identity_verified': source_verified, 'steps': steps,
            'terminal_accounting': all(s['terminal_accounting'] for s in steps),
            'pre': _phase(steps[:at]), 'post': _phase(steps[at:]),
            'optimizer': optimizer, 'eval': eval_results, 'eval_identity': eval_identity, 'eval_format': eval_format,
            'provenance': provenance}


def _comparable_config(config):
    # The effective config retains remote paths when a home is copied locally.
    prefix = str(Path(config['save']).parent.parent)

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key != 'RELOCATION_ARM'}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            return value.replace(prefix + '/', '<RUN_HOME>/')
        return value

    return normalize(config)


def compare_runs(healthy, faulty, repaired):
    """Apply fixed causal separation checks; report effects without significance claims."""
    runs = [audit_run(home) for home in (healthy, faulty, repaired)]
    _require([r['arm'] for r in runs] == list(ARMS), 'Comparison requires healthy/faulty/repaired in order')
    baseline = runs[0]
    for run in runs[1:]:
        for key in ('steps', 'relocate_at', 'model_revision', 'source_revision', 'model_files',
                    'case_sha256', 'manifest_sha256', 'eval_sha256', 'plugin_files', 'native_checkpoint_files'):
            _require(run['inputs'].get(key) == baseline['inputs'].get(key), f'INPUTS {key} differs across arms')
        _require(_comparable_config(run['config']) == _comparable_config(baseline['config']),
                 'effective config/budgets differ across arms')
        _require(run['packages'] == baseline['packages'], 'Runtime package versions differ across arms')
        _require(run['source_ids'] == baseline['source_ids'], 'Source IDs/intended sampling differ across arms')
        _require(run['eval_identity'] == baseline['eval_identity'], 'Fixed eval identities differ across arms')
        if run['provenance']['complete'] and baseline['provenance']['complete']:
            for key in ('source_receipt_sha256', 'conversion_receipt_sha256'):
                _require(run['provenance'][key] == baseline['provenance'][key], f'Provenance {key} differs across arms')
    unmet = []
    for run in runs:
        at = run['inputs']['relocate_at']
        for step in run['steps']:
            if run['arm'] == 'faulty' and step['step'] >= at:
                _require(step['abort_count'] > 0 and step['hard_fraction'] == 0,
                         'faulty post phase requires missing reads and zero admitted hard share')
            elif run['arm'] == 'faulty':
                _require(step['abort_count'] == 0 and step['hard_fraction'] > 0,
                         f"{run['arm']} control/pre phase requires zero aborts and positive hard share")
            else:
                _require(step['abort_count'] == 0, f"{run['arm']} control phase requires zero aborts")
                # Exact count equality avoids a tolerance that could admit
                # scheduler-biased control batches or hide them in phase means.
                if 2 * step['hard_count'] != step['admitted_count']:
                    unmet.append(f"{run['arm']} step {step['step']} hard_fraction={step['hard_fraction']}; "
                                 'the intended alternating control batch requires exactly 0.5.')
    deltas = {f"{r['arm']}_reward_delta": r['post']['reward_mean'] - r['pre']['reward_mean'] for r in runs}
    for control in ('healthy', 'repaired'):
        deltas[f'faulty_minus_{control}_delta'] = deltas['faulty_reward_delta'] - deltas[f'{control}_reward_delta']
    # Pool raw admitted reward sums and sample counts, never per-arm means or
    # the historical difficulty labels' source accuracies.
    pooled_pre = {}
    for difficulty in ('easy', 'hard'):
        count = sum(r['pre'][f'{difficulty}_count'] for r in runs)
        total = sum(r['pre'][f'{difficulty}_reward_sum'] for r in runs)
        pooled_pre.update({f'{difficulty}_count': count, f'{difficulty}_reward_sum': total,
                           f'{difficulty}_reward_mean': total / count if count else None})
    gap = (pooled_pre['easy_reward_mean'] - pooled_pre['hard_reward_mean']
           if pooled_pre['easy_count'] and pooled_pre['hard_count'] else None)
    pooled_pre['easy_minus_hard_reward'] = gap
    if any(r['eval_format'] != 'structured' for r in runs):
        unmet.append('Full three-arm completion requires phase-explicit structured eval snapshots; legacy files are pipeline evidence only.')
    if any(not r['terminal_accounting'] for r in runs):
        unmet.append('Full three-arm completion requires complete terminal accounting for every requested sample.')
    if any(not r['provenance']['complete'] for r in runs):
        unmet.append('Full three-arm completion requires native checkpoint and source/data provenance receipts.')
    if deltas['faulty_reward_delta'] <= 0:
        unmet.append('Observed faulty survivor reward did not increase; no values were substituted.')
    for control in ('healthy', 'repaired'):
        key = f'faulty_minus_{control}_delta'
        if deltas[key] <= 0:
            unmet.append(f'{key}={deltas[key]}; the faulty reward rise must exceed the {control} rise.')
    if gap is None or gap <= 0:
        unmet.append(f'pooled_pre easy_minus_hard_reward={gap}; measured admitted easy reward must exceed hard reward.')
    complete = not unmet
    return {'valid': True, 'causal_complete': complete,
            'status': 'causal_complete' if complete else 'causal_incomplete',
            'synthetic_fault_injection': True, **deltas,
            'pooled_pre': pooled_pre, 'unmet': unmet,
            'runs': {r['arm']: r for r in runs}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--healthy', type=Path)
    parser.add_argument('--faulty', type=Path)
    parser.add_argument('--repaired', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    arms = (args.healthy, args.faulty, args.repaired)
    if (args.run is not None and any(arms)) or (args.run is None and not all(arms)):
        parser.error('choose --run HOME or all of --healthy HOME --faulty HOME --repaired HOME')
    try:
        result = audit_run(args.run) if args.run else compare_runs(*arms)
        code = 0 if args.run or result['causal_complete'] else 1
    except ValueError as exc:
        result = {'valid': False, 'causal_complete': False, 'status': 'invalid', 'error': str(exc)}
        code = 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(rendered)
    sys.stdout.write(rendered)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
