"""Independent, stdlib-only continuation audit; never executes the submitted loader.

The caller supplies the sealed scene separately. Native evidence is trusted as
producer evidence, not as arbitrary authenticated claims. Reuse only the existing
independent auditor's JSON, accounting and checkpoint-schema utilities; no model,
producer summarizer, checkpoint deserialization, or implicit repr serialization.
"""
from collections import Counter, defaultdict
import argparse
import importlib.util
import json
from pathlib import Path


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
A = _module('incident_accounting', HERE / 'accounting.py')
S = _module('incident_scene_bytes', HERE / 'scene.py')
require = A._require
ROLES, STEPS = ('actor', 'critic'), [4, 5, 6, 7]


def _equal(actual, expected, context):
    require(type(actual) is type(expected) and actual == expected, context + ': value differs')


def _same_config(actual, expected):
    """Allow equivalent JSON numbers, but never coerce booleans, including nested flags."""
    if type(actual) in (int, float) and type(expected) in (int, float):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_same_config(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_same_config(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _absolute(value, label):
    path = Path(A._text(value, label))
    require(path.is_absolute(), label + ' must be an absolute path')
    return path.resolve()


def _config(config, base, scene, home):
    config, base = dict(config), dict(base)
    # Pinned Megatron declares exactly these skip flags with default=None.
    for values in (config, base):
        for key in ('no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng'):
            if key in values and values[key] is None:
                values[key] = False
    fixed = dict(start_rollout_id=None, finetune=False, no_load_optim=False, no_load_rng=False,
        no_save_optim=False, no_save_rng=False, debug_train_only=False, debug_rollout_only=False,
        debug_disable_optimizer=False, debug_skip_weight_update=False, async_save=False,
        override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=True,
        load_debug_rollout_data=None, pretrained_checkpoint=None, ckpt_step=None,
        ckpt_format='torch_dist', megatron_to_hf_mode='raw', partial_rollout=False, use_critic=True,
        num_rollout=8, rollout_batch_size=8, over_sampling_batch_size=8, n_samples_per_prompt=2,
        global_batch_size=16, micro_batch_size=1, seed=20261031, rollout_seed=20261031,
        train_backend='megatron', advantage_estimator='ppo', optimizer='adam', rm_type='math',
        update_weights_interval=1, num_critic_only_steps=0, eval_interval=4, save_interval=4,
        n_samples_per_eval_prompt=1, eval_temperature=0.0, save_retain_interval=4,
        data_source_path='rollout.BackedMathSource', rollout_function_path='rollout.generate_rollout',
        eval_function_path='rollout.generate_rollout', custom_generate_function_path='rollout.generate',
        dynamic_sampling_filter_path='miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted')
    fixed.update(load=str(scene / 'checkpoints/actor'), critic_load=str(scene / 'checkpoints/critic'),
        save=str(home / 'run/actor'), critic_save=str(home / 'run/critic'),
        prompt_data=[str(scene / 'dataset/eval.jsonl')], eval_prompt_data=['fixed', str(scene / 'dataset/eval.jsonl')])
    for key, value in fixed.items():
        require(key in config, 'Missing config ' + key)
        if type(value) is float:
            require(_same_config(config[key], value), 'config ' + key + ': value differs')
        else:
            _equal(config[key], value, 'config ' + key)
    hf = _absolute(config.get('hf_checkpoint'), 'HF checkpoint')
    if 'tokenizer_model' in base or 'tokenizer_model' in config:
        require('tokenizer_model' in config, 'Missing tokenizer model config')
        if config['tokenizer_model'] is None:
            require(base.get('tokenizer_model') is None, 'Tokenizer model was removed')
        else:
            _equal(_absolute(config['tokenizer_model'], 'Tokenizer model'), hf, 'Tokenizer/HF model root')
    require(_absolute(config.get('ref_load'), 'Reference load') in
            {_absolute(base.get('ref_load'), 'Original reference load'), scene / 'checkpoints/actor'},
            'Reference load must retain the original root or use the sealed actor checkpoint')
    # Only these driver-level differences are inherent in a continuation.
    changes = {'load', 'critic_load', 'save', 'critic_save', 'prompt_data', 'eval_prompt_data',
        'start_rollout_id', 'finetune', 'no_load_optim', 'no_load_rng', 'save_retain_interval',
        'use_checkpoint_opt_param_scheduler', 'data_source_path', 'rollout_function_path',
        'eval_function_path', 'custom_generate_function_path', 'train_env_vars',
        'save_debug_rollout_data', 'tensorboard_dir', 'eval_datasets', 'hf_checkpoint', 'tokenizer_model', 'ref_load'}
    for key in (base.keys() | config.keys()) - changes:
        require(key in base and key in config, 'Training config field added/deleted: ' + key)
        require(_same_config(config[key], base[key]), 'Training config ' + key + ': value differs')
    if 'eval_datasets' in base or 'eval_datasets' in config:
        # Native args serialize EvalDatasetConfig as text. Compare it opaquely,
        # allowing only the dataset path move; never import/eval its repr.
        before = json.dumps(base.get('eval_datasets'), sort_keys=True)
        after = json.dumps(config.get('eval_datasets'), sort_keys=True)
        _equal(after, before.replace(base['eval_prompt_data'][1], fixed['eval_prompt_data'][1]), 'eval_datasets')
    env = config['train_env_vars']
    for key, value in dict(RELOCATION_DATA=str(scene / 'dataset'), RELOCATION_OUTPUT=str(home / 'run'),
        RELOCATION_ARM='incident', RELOCATION_STEPS='8', RELOCATION_AT='4',
        BENCH_RESTORE_ROOT=str(scene / 'checkpoints'), BENCH_RESTORE_ITERATION='3').items():
        _equal(env.get(key), value, 'config environment ' + key)


def _model_assets(scene, receipt, config):
    """Bind HF bytes to sealed INPUTS or the engine/wrapper's trusted snapshot."""
    source, base = receipt['source']['inputs'], receipt['source']['parameters']
    root = _absolute(config['hf_checkpoint'], 'HF checkpoint')
    moved = root != _absolute(base['hf_checkpoint'], 'Original HF checkpoint')
    pins, origin = source.get('model_files'), 'scene'
    for snapshot_path in (HERE / 'snapshot.json', HERE.parent / 'snapshot.json'):
        if not snapshot_path.is_file():
            continue
        snapshot = A._load(snapshot_path)
        if 'model_files' in snapshot:
            _equal(snapshot.get('scene_sha256'), A._hash(scene / 'SCENE.json'), 'Model pins scene version')
            for key in ('model_revision', 'source_revision'):
                if key in snapshot:
                    _equal(snapshot[key], source[key], 'Model pins ' + key)
            if pins is not None:
                _equal(snapshot['model_files'], pins, 'Snapshot/sealed model file pins')
            else:
                pins, origin = snapshot['model_files'], 'snapshot'
    if pins is None:
        require(not moved, 'Relocated HF assets require scene-bound model_files pins')
        return dict(relocated=False, pins=None, files_rehashed=0)
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                'chat_template.jinja', 'model.safetensors'}
    require(isinstance(pins, dict) and required <= pins.keys(), 'Incomplete model file pins')
    for name, digest in pins.items():
        S._relative(name)
        A._digest(digest, 'Model file pin ' + name)
    files = S._inventory(S._root(root))
    require(all(info['size'] > 0 for info in files.values()), 'Empty model asset')
    _equal({name: info['sha256'] for name, info in files.items()}, pins, 'HF asset inventory SHA256/file-set')
    return dict(relocated=moved, pins=origin, files_rehashed=len(files))


def _attempt_trace(run, events, scene):
    path = run / 'read_attempts.jsonl'
    if not path.exists():
        return {'present': False}
    reads = {(r['step'], r['index']): r for r in events if r['kind'] == 'read'}
    attempts, sequences = A._lines(path), defaultdict(list)
    totals = {str(s): dict(attempts=0, missing=0, final_reads=sum(k[0] == s for k in reads)) for s in STEPS}
    for row in attempts:
        key = (A._integer(row.get('step'), 'Attempt step'), A._integer(row.get('index'), 'Attempt index', 64))
        require(key in reads, 'Attempt trace has no matching final read')
        read, previous = reads[key], sequences[key]
        _equal(row.get('id'), read['id'], 'Attempt logical ID')
        attempt = A._integer(row.get('attempt'), 'Attempt sequence', 1)
        require(attempt == len(previous) + 1 and attempt <= 3, 'Attempt sequence must be contiguous within 1..3')
        require(not previous or previous[-1]['outcome'] == 'missing', 'Attempt follows an already successful read')
        require(row.get('outcome') in ('missing', 'ok'), 'Unknown attempt outcome')
        if row['outcome'] == 'ok':
            _equal(row.get('sha256'), read.get('sha256'), 'Attempt content hash')
            resolved = A._text(row.get('path'), 'Resolved attempt path')
            payload = S._path(scene / 'dataset', Path(resolved).as_posix())
            _equal(A._hash(payload), read.get('sha256'), 'Resolved attempt bytes')
        else:
            require(row.get('sha256') is None and row.get('path') is None, 'Missing attempt claims resolved content')
        previous.append(row)
        totals[str(key[0])]['attempts'] += 1
        totals[str(key[0])]['missing'] += row['outcome'] == 'missing'
    require(sequences.keys() == reads.keys(), 'Attempt trace omits final reads')
    for key, rows in sequences.items():
        _equal(rows[-1]['outcome'], reads[key]['outcome'], 'Attempt/final read outcome')
        require(rows[-1]['outcome'] == 'ok' or len(rows) == 3, 'Final missing read did not exhaust three attempts')
    return dict(present=True, attempts=len(attempts), missing=sum(v['missing'] for v in totals.values()),
                final_reads=len(reads), steps=totals)


def _events(run, manifest, selection, expected, config, scene):
    events = A._lines(run / 'events.jsonl')
    first = events[0]
    for key, value in dict(kind='datasource_restored', step=3, cursor=32, sample_index=64).items():
        _equal(first.get(key), value, 'datasource restore ' + key)
    allowed = {'rollout_begin', 'read', 'terminal', 'admitted', 'rollout_end'}
    for event in events:
        _equal(event.get('arm'), 'incident', 'event arm')
    for event in events[1:]:
        require(type(event.get('step')) is int and event['step'] in STEPS and event.get('kind') in allowed,
                'Event outside continuation steps/kinds')
    require([e['step'] for e in events] == sorted(e['step'] for e in events), 'Event timesteps rewind')
    by_id = {r['id']: r for r in manifest}
    labels = {r['id']: r['label_sha256'] for r in selection}
    next_index, summaries = 64, []
    for step in STEPS:
        rows = [e for e in events if e['step'] == step]
        require(rows and rows[0]['kind'] == 'rollout_begin' and rows[-1]['kind'] == 'rollout_end'
                and sum(e['kind'] == 'rollout_begin' for e in rows) == 1, 'Missing/duplicate rollout boundary')
        _equal(rows[0].get('cursor'), next_index // 2, 'rollout cursor')
        summary = A.audit_step(rows, step)
        require(summary['terminal_accounting'], 'Missing terminal accounting')
        terminals = [e for e in rows if e['kind'] == 'terminal']
        indices = sorted(e['index'] for e in terminals)
        require(indices == list(range(next_index, next_index + len(indices))) and len(indices) % 2 == 0,
                'Requested samples missing, duplicated, or rewound')
        groups, admitted = {}, defaultdict(list)
        for event in rows:
            if event['kind'] not in ('read', 'terminal', 'admitted'):
                continue
            source = by_id.get(event['id'])
            require(source is not None and event['difficulty'] == source['difficulty'], 'Sample identity differs from scene')
            group = A._integer(event.get('group_index'), 'group index', 32)
            require(event['index'] // 2 == group, 'Native sample/group indices disagree')
            require(groups.setdefault(group, event['id']) == event['id'], 'Native group contains replacement IDs')
            if event['kind'] == 'read' and event['outcome'] == 'ok':
                _equal(event.get('sha256'), source['sha256'], 'Read content hash')
                # path is the original manifest reference, NOT the resolved path.
            if event['kind'] in ('terminal', 'admitted') and event['status'] in ('completed', 'truncated'):
                _equal(A._content_hash(event['label']), labels[event['id']], 'Source label hash')
                require(bool(event['response']) and event['response_length'] <= config['rollout_max_response_len'],
                        'Missing/over-budget generated response')
            if event['kind'] == 'admitted':
                admitted[group].append(event)
        require(len(admitted) == 8 and all(sorted(e['index'] for e in rows) == [g * 2, g * 2 + 1]
                for g, rows in admitted.items()), 'Incomplete admitted logical groups')
        counts = Counter(e['id'] for rows in admitted.values() for e in rows)
        require(counts == Counter({k: v * 2 for k, v in Counter(expected[str(step)]).items()}),
                'Scheduled logical IDs were replaced or silently lost')
        next_index += len(indices)
        summaries.append(summary)
    state = A._load(run / 'actor/rollout/data_state_7.json')
    for key, value in dict(step=7, arm='incident', cursor=next_index // 2,
                           group_index=next_index // 2, sample_index=next_index).items():
        _equal(state.get(key), value, 'Final datasource ' + key)
    return summaries, _attempt_trace(run, events, scene)


def _snapshot(value, counter):
    for key in ('model_sha256', 'optimizer_sha256'):
        A._digest(value[key], 'snapshot ' + key)
    for key in ('model_tensors', 'model_elements', 'optimizer_tensors', 'optimizer_elements'):
        A._integer(value[key], 'snapshot ' + key, 1)
    require(value['optimizer_steps'] and all(A._number(s, 'Adam counter') == counter
            for s in value['optimizer_steps']), 'Reset/missing Adam state')
    for key in ('exp_avg', 'exp_avg_sq'):
        A._integer(value['moment_elements'][key], 'Adam moment ' + key, 1)


def _optimizer(run, scene):
    records = [r for p in sorted((run / 'optimizer_evidence').glob('*.jsonl')) for r in A._lines(p)]
    require(records and {r.get('role') for r in records} == set(ROLES), 'Missing actor/critic optimizer evidence')
    result = {}
    for role in ROLES:
        rows = [r for r in records if r['role'] == role]
        expected = ['restore_loaded'] + ['optimizer_step', 'restore_optimizer_step'] * 4 + ['restore_saved']
        require([r.get('event') for r in rows] == expected, 'Missing/duplicate/unordered optimizer events: ' + role)
        pid = A._integer(rows[0].get('pid'), 'Worker PID', 1)
        for row in rows:
            _equal(row.get('rank'), 0, 'Worker rank')
            _equal(row.get('pid'), pid, 'Worker PID')
        loaded, saved = rows[0], rows[-1]
        for key, value in dict(loaded_iteration=3, next_rollout_id=4, payload_exact=True,
            scheduler_exact=True, rng_exact=True, load_root=str(scene / 'checkpoints' / role),
            scheduler_source='common.pt:opt_param_scheduler', rng_source='DCP:rng_state').items():
            _equal(loaded.get(key), value, role + ' restore ' + key)
        require(A._number(loaded.get('scheduler_num_steps'), 'scheduler progress') == 64, 'Scheduler progress reset')
        for key in ('scheduler_sha256', 'rng_sha256'):
            A._digest(loaded.get(key), key)
        for offset, step in enumerate(STEPS):
            normal, counter = rows[1 + offset * 2:3 + offset * 2]
            for row in (normal, counter):
                _equal(row.get('rollout_id'), step, 'Optimizer rollout')
                _equal(row.get('step_id'), 0, 'Optimizer step_id')
            for key in ('update_successful', 'grad_norm_finite', 'sampled_finite', 'sampled_changed'):
                _equal(normal.get(key), True, 'Optimizer ' + key)
            tensors = A._integer(normal.get('sampled_tensors'), 'sampled tensors', 1)
            elements = A._integer(normal.get('sampled_elements'), 'sampled elements', 1)
            changed = A._integer(normal.get('changed_tensors'), 'changed tensors', 1)
            require(changed <= tensors and normal.get('sampled_bytes') == elements * 4
                    and A._number(normal.get('grad_norm'), 'grad norm') >= 0
                    and A._number(normal.get('max_delta'), 'parameter delta') > 0, 'Invalid gradient/update evidence')
            _equal(counter.get('expected_optimizer_step'), step + 1, 'Expected Adam counter')
            require(counter.get('optimizer_steps') and all(A._number(s, 'Adam counter') == step + 1
                    for s in counter['optimizer_steps']), 'Adam counter did not advance')
        for key, value in dict(saved_iteration=7, step_delta=4, payload_exact=True, save_root=str(run / role)).items():
            _equal(saved.get(key), value, role + ' saved ' + key)
        before, after = loaded['snapshot'], saved['snapshot']
        _snapshot(before, 4)
        _snapshot(after, 8)
        for key in ('model_sha256', 'optimizer_sha256'):
            require(before[key] != after[key], role + ' final state did not change: ' + key)
        require(len(before['optimizer_steps']) == len(after['optimizer_steps']), 'Adam state count changed')
        for key in ('model_tensors', 'model_elements', 'optimizer_tensors', 'optimizer_elements', 'moment_elements',
                    'ignored_storage_padding_tensors', 'ignored_storage_padding_elements'):
            _equal(after[key], before[key], 'State layout ' + key)
        result[role] = dict(steps=STEPS, restored_optimizer_step=4, saved_optimizer_step=8, payload_exact=True)
    return result


def _checkpoints(home, fresh):
    outputs, rehashed = A._load(home / 'output_checkpoints.json'), {}
    require(set(outputs) == set(ROLES), 'Missing output checkpoint roles')
    for role, record in outputs.items():
        files = A._checkpoint(record, '7', True)
        require('common.pt' in files, 'Missing checkpoint common state')
        root = home / 'run' / role
        tracker = root / 'latest_checkpointed_iteration.txt'
        if fresh or tracker.exists():
            _equal(tracker.read_text().strip(), '7', 'Output checkpoint tracker')
        present = []
        for name, info in files.items():
            path = S._path(root, record['selected'] + '/' + name)
            present.append(path.is_file())
            if fresh or path.exists():
                _equal(S._info(path), info, 'Output checkpoint bytes ' + role + '/' + name)
        rehashed[role] = all(present)
    return rehashed


def _evaluation(run, scene, config):
    require({p.name for p in run.glob('eval_*.json')} == {'eval_after_step_0000007.json'},
            'Expected only the final step-7 evaluation')
    expected = A._lines(scene / 'dataset/eval.jsonl')
    output = A._load(run / 'eval_after_step_0000007.json')
    require(set(output) == {'fixed'} and len(expected) == 16, 'Expected fixed 16-sample evaluation')
    fixed = output['fixed']
    require(all(isinstance(fixed.get(k), list) and len(fixed[k]) == 16 for k in ('samples', 'rewards', 'truncated')),
            'Incomplete final evaluation')
    for index, (sample, source) in enumerate(zip(fixed['samples'], expected)):
        require(isinstance(sample, dict), 'Evaluation requires structured JSON samples')
        _equal(sample.get('index'), index, 'Evaluation sample index')
        _equal(sample.get('label'), source['label'], 'Evaluation label')
        for key in ('source_line', 'difficulty'):
            _equal(sample['metadata'].get(key), source['metadata'][key], 'Evaluation identity ' + key)
        prompt = sample.get('prompt')
        if isinstance(prompt, str):
            # Native chat templating changes representation, but must retain the task text in order.
            for message in source['prompt']:
                pos = prompt.find(message['content'])
                require(pos >= 0, 'Evaluation prompt differs from fixed task')
                prompt = prompt[pos + len(message['content']):]
        else:
            _equal(prompt, source['prompt'], 'Evaluation prompt')
        reward = A._reward(sample.get('reward'))
        require(A._reward(fixed['rewards'][index]) == reward, 'Evaluation reward mismatch')
        require(sample.get('status') in ('completed', 'truncated'), 'Evaluation incomplete')
        _equal(fixed['truncated'][index], sample['status'] == 'truncated', 'Evaluation truncation')
        A._text(sample.get('response'), 'Evaluation response')
        require(A._integer(sample.get('response_length'), 'Evaluation length', 1) <= config['eval_max_response_len'],
                'Evaluation exceeds generation budget')
    return dict(step=7, count=16, reward_mean=sum(fixed['rewards']) / 16)


def verify_resume(scene, home, fresh=True) -> dict:
    """Return strict JSON proof; fresh=False permits missing archived output bytes only."""
    proof = dict(schema_version=3, score=False, errors=[])
    try:
        scene, home = Path(scene).resolve(strict=True), Path(home).resolve(strict=True)
        require(not home.is_relative_to(scene) and not scene.is_relative_to(home), 'Scene/output roots overlap')
        receipt = S.verify_scene(scene)
        before = A._hash(scene / 'SCENE.json')
        inputs, result = A._load(home / 'INPUTS.json'), A._load(home / 'RESULT.json')
        for key, value in dict(schema_version=2, checkpoint_iteration=3, start_step=4, final_step=7,
                               samples_per_prompt=2, scene_sha256=before).items():
            _equal(inputs.get(key), value, 'INPUTS ' + key)
        for key in ('model_revision', 'source_revision'):
            _equal(inputs.get(key), receipt['source']['inputs'][key], 'INPUTS ' + key)
        _equal(inputs.get('loader_sha256'), A._hash(home / 'submitted/loader.py'), 'Submitted loader hash')
        for key, value in dict(returncode=0, original_scene_unchanged=True, restored_iteration=3,
                               completed_steps=STEPS, scene_sha256=before).items():
            _equal(result.get(key), value, 'RESULT ' + key)
        require(receipt['sample_cursor'] == 32 and receipt['sample_index'] == 64, 'Scene cursor differs')
        manifest = A._load(scene / 'dataset/manifest.json')['rows']
        expected = {str(s): [manifest[(32 + (s - 4) * 8 + i) % len(manifest)]['id'] for i in range(8)] for s in STEPS}
        _equal(inputs.get('expected_groups'), expected, 'INPUTS expected_groups')
        run = home / 'run'
        config = A._load(run / 'effective_config.json')
        _config(config, receipt['source']['parameters'], scene, home)
        model_assets = _model_assets(scene, receipt, config)
        returned = A._load(run / 'TRAINING_RETURNED.json')
        for key, value in dict(training_returned=True, arm='incident', physical_gpus=3).items():
            _equal(returned.get(key), value, 'TRAINING_RETURNED ' + key)
        steps, trace = _events(run, manifest, A._load(scene / 'provenance/selection.json')['rows'], expected, config, scene)
        proof.update(expected_groups=expected, steps=steps, admitted_count=sum(s['admitted_count'] for s in steps),
                     read_attempt_trace=trace, model_assets=model_assets,
                     optimizer=_optimizer(run, scene), checkpoint_payloads_rehashed=_checkpoints(home, fresh),
                     evaluation=_evaluation(run, scene, config))
        _equal(S.verify_scene(scene), receipt, 'Original complete scene after audit')
        _equal(A._hash(scene / 'SCENE.json'), before, 'Original scene receipt after audit')
        proof.update(score=True, original_scene_unchanged=True, scene_sha256=before, fresh=fresh)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError) as error:
        proof['errors'].append(str(error))
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--no-fresh', action='store_true', help='Allow archived output checkpoint files to be absent')
    args = parser.parse_args()
    result = verify_resume(args.scene, args.home, fresh=not args.no_fresh)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result['score'] else 1)
