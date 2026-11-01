"""Isolated SYNTHETIC schema fixtures; checkpoint bytes are NOT loadable models."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
TRUSTED_BUNDLE = None


@pytest.fixture(autouse=True)
def isolated_trusted_bundle(tmp_path, monkeypatch):
    """Keep synthetic/config-only tests independent of the published model pins."""
    root = tmp_path / 'synthetic_trusted_bundle/incident'
    root.mkdir(parents=True)
    for name in ('verify_resume.py', 'accounting.py', 'scene.py'):
        shutil.copyfile(HERE / name, root / name)
    monkeypatch.setitem(globals(), 'TRUSTED_BUNDLE', root)
    return root


def module(name):
    root = TRUSTED_BUNDLE if name == 'verify_resume' else HERE
    assert root is not None, 'Verifier tests require their isolated trusted bundle'
    path = root / (name + '.py')
    assert path.is_file(), name + ' implementation is missing'
    spec = importlib.util.spec_from_file_location('test_incident_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): {'size': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def snapshot(step):
    return dict(model_sha256=canonical(['model', step]), optimizer_sha256=canonical(['adam', step]),
                model_tensors=2, model_elements=8, optimizer_tensors=6, optimizer_elements=20,
                optimizer_steps=[float(step)] * 2, moment_elements={'exp_avg': 8, 'exp_avg_sq': 8},
                ignored_storage_padding_tensors=0, ignored_storage_padding_elements=0)


class Fixture:
    def __init__(self, root):
        self.scene, self.home = root / 'scene', root / 'continuation'
        prefix, data = root / 'prefix', root / 'prefix/dataset'
        self.rows, self.evaluation, selected = [], [], []
        self.payloads = {}
        for i in range(80):
            split = 'train' if i < 64 else 'eval'
            difficulty = 'easy' if i < 48 or 64 <= i < 72 else 'hard'
            identifier = f'synthetic-{i:03d}'
            payload = dict(id=identifier, prompt=[{'role': 'user', 'content': f'SYNTHETIC {i}+0?'}],
                           label=str(i), metadata=dict(difficulty=difficulty, source_line=i + 1,
                                                      historical_accuracy=float(difficulty == 'easy')))
            self.payloads[identifier] = payload
            row = dict(id=identifier, split=split, difficulty=difficulty, source_line=i + 1,
                       uuid=f'SYNTHETIC_{i}', file='synthetic.jsonl', row_index_0based=i,
                       historical_accuracy=payload['metadata']['historical_accuracy'],
                       public_record_sha256=canonical(payload), prompt_sha256=canonical(payload['prompt']),
                       label_sha256=canonical(payload['label']),
                       content_sha256=canonical({k: payload[k] for k in ('prompt', 'label')}))
            selected.append(dict(row, source_row_id=row['uuid'], public_file=row['file']))
            if split == 'train':
                name = f'live/{difficulty}/{identifier}.json'
                write(data / name, payload)
                self.rows.append(dict(id=identifier, split=split, difficulty=difficulty,
                                      source_line=i + 1, path=name, sha256=digest(data / name)))
            else:
                self.evaluation.append(payload)
        write(data / 'manifest.json', dict(schema_version=1, rows=self.rows))
        lines(data / 'eval.jsonl', self.evaluation)
        files = inventory(data)
        source = dict(repo='SYNTHETIC/schema-only', revision='a' * 40)
        counts = {split: dict(Counter(r['difficulty'] for r in selected if r['split'] == split))
                  for split in ('train', 'eval')}
        selection = dict(schema_version=1, rows=selected, source=source, expected_counts=counts,
                         expected_runtime_sha256={k: v['sha256'] for k, v in files.items()})
        write(root / 'selection.json', selection)
        write(data / 'CASE.json', dict(schema_version=1, synthetic_fault_injection=True, source=source, rows=selected))
        write(data / 'PUBLIC_PROVENANCE.json', dict(schema_version=1, synthetic_fault_injection=True,
              source=source, inputs={'selection_sha256': digest(root / 'selection.json')},
              case_sha256=digest(data / 'CASE.json'), runtime_sha256=selection['expected_runtime_sha256'],
              runtime_sizes={k: v['size'] for k, v in files.items()}, verification={'counts': counts}))
        self.config = dict(num_rollout=8, rollout_batch_size=8, over_sampling_batch_size=8,
            n_samples_per_prompt=2, global_batch_size=16, micro_batch_size=1, seed=20261031,
            rollout_seed=20261031, train_backend='megatron', advantage_estimator='ppo', optimizer='adam',
            rm_type='math', use_critic=True, update_weights_interval=1, num_critic_only_steps=0,
            eval_interval=4, n_samples_per_eval_prompt=1, eval_temperature=0.0, save_interval=4,
            rollout_max_response_len=32768, eval_max_response_len=32768, lr=1e-6, critic_lr=5e-6,
            finetune=True, no_load_optim=True, no_load_rng=True, start_rollout_id=0,
            no_save_optim=False, no_save_rng=None, debug_train_only=False, debug_rollout_only=False,
            debug_disable_optimizer=False, debug_skip_weight_update=False, async_save=False,
            override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False,
            partial_rollout=False, load_debug_rollout_data=None, pretrained_checkpoint=None,
            ckpt_step=None, ckpt_format='torch_dist', megatron_to_hf_mode='raw',
            save_retain_interval=100000, skip_eval_before_train=False,
            hf_checkpoint='/synthetic/model', tokenizer_model='/synthetic/model', ref_load='/synthetic/reference',
            data_source_path='relocation_rollout.BackedMathSource',
            rollout_function_path='relocation_rollout.generate_rollout',
            eval_function_path='relocation_rollout.generate_rollout',
            custom_generate_function_path='relocation_rollout.generate',
            dynamic_sampling_filter_path='miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted')
        self.config.update(load=str(prefix / 'run/actor'), critic_load=str(prefix / 'run/critic'),
            save=str(prefix / 'run/actor'), critic_save=str(prefix / 'run/critic'),
            prompt_data=[str(data / 'eval.jsonl')], eval_prompt_data=['fixed', str(data / 'eval.jsonl')],
            train_env_vars={})
        write(prefix / 'run/effective_config.json', self.config)
        base_inputs = dict(arm='healthy', steps=8, relocate_at=4, source_revision='c' * 40,
                          model_revision='b' * 40, **{key: digest(data / name) for name, key in
                          [('manifest.json', 'manifest_sha256'), ('eval.jsonl', 'eval_sha256'), ('CASE.json', 'case_sha256')]})
        write(prefix / 'INPUTS.json', base_inputs)
        write(prefix / 'run/actor/rollout/data_state_3.json',
              dict(cursor=32, group_index=32, sample_index=64, step=3, arm='healthy'))
        for role in ('actor', 'critic'):
            self.checkpoint(prefix / 'run' / role, 3)
            lines(prefix / f'run/optimizer_evidence/{role}.jsonl',
                  [dict(event='optimizer_step', role=role, rollout_id=s, update_successful=True,
                        synthetic_fixture=True) for s in range(4)])
        lines(prefix / 'run/events.jsonl', [dict(kind=k, step=s, synthetic_fixture=True)
              for s in range(4) for k in ('rollout_begin', 'rollout_end')])
        module('scene').create_scene(prefix, self.scene, selection_path=root / 'selection.json')
        self.config.update(finetune=False, no_load_optim=False, no_load_rng=False,
            start_rollout_id=None, use_checkpoint_opt_param_scheduler=True, save_retain_interval=4,
            load=str(self.scene / 'checkpoints/actor'), critic_load=str(self.scene / 'checkpoints/critic'),
            save=str(self.home / 'run/actor'), critic_save=str(self.home / 'run/critic'),
            prompt_data=[str(self.scene / 'dataset/eval.jsonl')],
            eval_prompt_data=['fixed', str(self.scene / 'dataset/eval.jsonl')])
        for key in ('data_source_path', 'rollout_function_path', 'eval_function_path', 'custom_generate_function_path'):
            self.config[key] = self.config[key].replace('relocation_rollout.', 'rollout.')
        self.config['train_env_vars'] = dict(RELOCATION_DATA=str(self.scene / 'dataset'),
            RELOCATION_OUTPUT=str(self.home / 'run'), RELOCATION_ARM='incident', RELOCATION_STEPS='8',
            RELOCATION_AT='4', BENCH_RESTORE_ROOT=str(self.scene / 'checkpoints'), BENCH_RESTORE_ITERATION='3')
        write(self.home / 'run/effective_config.json', self.config)
        loader = self.home / 'submitted/loader.py'
        loader.parent.mkdir(parents=True)
        loader.write_text('# SYNTHETIC fixture only; never executed by verifier\n')
        self.groups = {str(s): [r['id'] for r in self.rows[32 + (s - 4) * 8:40 + (s - 4) * 8]] for s in range(4, 8)}
        write(self.home / 'INPUTS.json', dict(schema_version=2, scene_sha256=digest(self.scene / 'SCENE.json'),
            checkpoint_iteration=3, start_step=4, final_step=7, samples_per_prompt=2,
            expected_groups=self.groups, loader_sha256=digest(loader), source_revision='c' * 40, model_revision='b' * 40))
        write(self.home / 'RESULT.json', dict(returncode=0, original_scene_unchanged=True,
            restored_iteration=3, completed_steps=[4, 5, 6, 7], scene_sha256=digest(self.scene / 'SCENE.json')))
        write(self.home / 'run/TRAINING_RETURNED.json', dict(training_returned=True, arm='incident', physical_gpus=3))
        outputs = {role: self.checkpoint(self.home / 'run' / role, 7) for role in ('actor', 'critic')}
        write(self.home / 'output_checkpoints.json', outputs)
        for role in ('actor', 'critic'):
            worker = dict(role=role, rank=0, pid=123, synthetic_fixture=True)
            evidence = [dict(worker, event='restore_loaded', loaded_iteration=3, next_rollout_id=4,
                load_root=str(self.scene / 'checkpoints' / role), payload_exact=True, snapshot=snapshot(4),
                scheduler_exact=True, rng_exact=True, scheduler_sha256='d' * 64, rng_sha256='e' * 64,
                scheduler_num_steps=64, scheduler_source='common.pt:opt_param_scheduler', rng_source='DCP:rng_state')]
            for step in range(4, 8):
                evidence.extend([dict(worker, event='optimizer_step', rollout_id=step, step_id=0,
                    update_successful=True, grad_norm_finite=True, sampled_finite=True, sampled_changed=True,
                    grad_norm=0.25, sampled_tensors=2, sampled_elements=8, sampled_bytes=32,
                    changed_tensors=2, max_delta=0.001),
                    dict(worker, event='restore_optimizer_step', rollout_id=step, step_id=0,
                         optimizer_steps=[float(step + 1)], expected_optimizer_step=step + 1)])
            evidence.append(dict(worker, event='restore_saved', saved_iteration=7, step_delta=4,
                                 save_root=str(self.home / 'run' / role), payload_exact=True, snapshot=snapshot(8)))
            lines(self.home / f'run/optimizer_evidence/{role}.jsonl', evidence)
        samples = [dict(index=i, prompt=p['prompt'], label=p['label'], metadata=p['metadata'],
                        response='SYNTHETIC wrong answer', response_length=4, status='completed', reward=0)
                   for i, p in enumerate(self.evaluation)]
        write(self.home / 'run/eval_after_step_0000007.json',
              {'fixed': dict(samples=samples, rewards=[0] * 16, truncated=[False] * 16)})
        self.events()

    @staticmethod
    def checkpoint(root, step):
        selected = root / f'iter_{step:07d}'
        selected.mkdir(parents=True)
        for name in ('.metadata', 'common.pt', '__0_0.distcp'):
            (selected / name).write_bytes(f'SYNTHETIC UNLOADABLE schema fixture {step}/{name}\n'.encode())
        (root / 'latest_checkpointed_iteration.txt').write_text(str(step) + '\n')
        return dict(iteration=str(step), selected=selected.name, referenced_files=inventory(selected),
                    state_keys=['model.synthetic', 'optimizer.synthetic'], restore_tested=False)

    def events(self, retry=False):
        producer = module('rollout')
        events = [dict(kind='datasource_restored', step=3, arm='incident', cursor=32, sample_index=64)]
        cursor = 32
        for step in range(4, 8):
            events.append(dict(kind='rollout_begin', step=step, arm='incident', cursor=cursor))
            reads, terminals, admitted = [], [], []
            plan = [(i, False) for i in self.groups[str(step)]]
            if retry and step == 4:
                plan.insert(0, (plan[0][0], True))
            for identifier, failed in plan:
                row = next(r for r in self.rows if r['id'] == identifier)
                for index in range(cursor * 2, cursor * 2 + 2):
                    identity = dict(step=step, arm='incident', id=identifier, index=index,
                                    group_index=cursor, difficulty=row['difficulty'])
                    read = dict(identity, kind='read', path=row['path'], outcome='missing' if failed else 'ok')
                    read.update({'errno': 2} if failed else {'sha256': row['sha256']})
                    terminal = dict(identity, kind='terminal', status='aborted' if failed else 'completed',
                        reward=None if failed else 0, response='' if failed else 'SYNTHETIC answer',
                        response_length=0 if failed else 2, label=None if failed else self.payloads[identifier]['label'])
                    reads.append(read); terminals.append(terminal)
                    events.extend([read, terminal])
                    if not failed:
                        admitted.append(dict(terminal, kind='admitted'))
                cursor += 1
            events.extend(admitted)
            metrics = producer.summarize(reads, admitted)
            metrics.update(producer.terminal_metrics(reads, terminals, [a['index'] for a in admitted],
                                                    [t['index'] for t in terminals]))
            events.append(dict(kind='rollout_end', step=step, arm='incident', metrics=metrics))
        lines(self.home / 'run/events.jsonl', events)
        write(self.home / 'run/actor/rollout/data_state_7.json',
              dict(cursor=cursor, group_index=cursor, sample_index=cursor * 2, step=7, arm='incident'))

    def edit(self, name, change, jsonl=False):
        path = self.home / name
        value = [json.loads(s) for s in path.read_text().splitlines()] if jsonl else json.loads(path.read_text())
        change(value)
        (lines if jsonl else write)(path, value)

    def trace(self, transient=False):
        """Explicit synthetic attempt records matching every final read event."""
        records = []
        for row in map(json.loads, (self.home / 'run/events.jsonl').read_text().splitlines()):
            if row['kind'] != 'read':
                continue
            outcomes = ['missing'] * 3 if row['outcome'] == 'missing' else ['ok']
            if transient and not records:
                outcomes = ['missing', 'missing', 'ok']
            for attempt, outcome in enumerate(outcomes, 1):
                records.append(dict(id=row['id'], index=row['index'], step=row['step'], attempt=attempt,
                    outcome=outcome, path=row['path'].replace('live/hard/', 'archive/hard/') if outcome == 'ok' else None,
                    sha256=row.get('sha256') if outcome == 'ok' else None))
        lines(self.home / 'run/read_attempts.jsonl', records)

    def verify(self, **kwargs):
        result = module('verify_resume').verify_resume(self.scene, self.home, **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def source_config(self, changes):
        """Reseal only this explicitly synthetic fixture's source configuration."""
        path = self.scene / 'run/effective_config.json'
        config = json.loads(path.read_text())
        config.update(changes)
        write(path, config)
        receipt_path = self.scene / 'SCENE.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['source']['parameters'] = config
        info = {'size': path.stat().st_size, 'sha256': digest(path)}
        receipt['files']['run/effective_config.json'] = info
        receipt['source']['files']['run/effective_config.json'] = info
        write(receipt_path, receipt)
        for name in ('INPUTS.json', 'RESULT.json'):
            self.edit(name, lambda value: value.update(scene_sha256=digest(receipt_path)))

    def source_inputs(self, changes):
        """Reseal explicitly synthetic INPUTS; never modify real scene artifacts."""
        path = self.scene / 'INPUTS.json'
        value = json.loads(path.read_text())
        value.update(changes)
        write(path, value)
        receipt_path = self.scene / 'SCENE.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['source']['inputs'] = value
        info = {'size': path.stat().st_size, 'sha256': digest(path)}
        receipt['files']['INPUTS.json'] = receipt['source']['files']['INPUTS.json'] = info
        write(receipt_path, receipt)
        for name in ('INPUTS.json', 'RESULT.json'):
            self.edit(name, lambda record: record.update(scene_sha256=digest(receipt_path)))

    def model_assets(self):
        root = self.home.parent / 'prepared/assets/hf-safetensors'
        root.mkdir(parents=True)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'special_tokens_map.json', 'model.safetensors', 'README.md'):
            (root / name).write_bytes(('SYNTHETIC UNLOADABLE model fixture ' + name).encode())
        self.edit('run/effective_config.json', lambda config: config.update(
            hf_checkpoint=str(root), tokenizer_model=str(root), ref_load=str(self.scene / 'checkpoints/actor')))
        return root, {name: info['sha256'] for name, info in inventory(root).items()}


@pytest.fixture
def case(tmp_path):
    return Fixture(tmp_path)


def test_exact_schedule_zero_reward_and_original_manifest_paths_pass(case):
    before = inventory(case.scene)
    result = case.verify()
    assert result['score'] is True, result
    assert result['admitted_count'] == 64
    assert result['expected_groups'] == case.groups
    assert inventory(case.scene) == before


def test_retry_is_allowed_without_replacement_or_silent_loss(case):
    case.events(retry=True)
    result = case.verify()
    assert result['score'] is True, result
    assert result['steps'][0]['abort_count'] == 2
    assert result['steps'][0]['read_attempts'] == 18


@pytest.mark.parametrize('damage', ['rewind', 'deleted_id', 'replacement', 'metric_mask', 'missing_terminal',
                                  'wrong_hash', 'wrong_label', 'migration', 'cursor', 'duplicate_admission'])
def test_event_contract_rejects_incomplete_or_inconsistent_evidence(case, damage):
    def change(rows):
        if damage == 'rewind':
            for r in rows:
                if r['step'] == 4: r['step'] = 0
        elif damage == 'deleted_id':
            rows[:] = [r for r in rows if r.get('id') != case.groups['4'][0]]
        elif damage == 'replacement':
            replacement = case.rows[0]
            for r in rows:
                if r.get('id') == case.groups['4'][0]:
                    r['id'] = replacement['id']
                    if r['kind'] == 'read': r.update(sha256=replacement['sha256'], path=replacement['path'])
                    else: r['label'] = case.payloads[replacement['id']]['label']
        elif damage == 'metric_mask':
            next(r for r in rows if r['kind'] == 'rollout_end')['metrics']['data/read_attempts'] = 1
        elif damage == 'missing_terminal':
            rows.remove(next(r for r in rows if r['kind'] == 'terminal'))
        elif damage == 'wrong_hash': next(r for r in rows if r['kind'] == 'read')['sha256'] = 'f' * 64
        elif damage == 'wrong_label':
            for r in rows:
                if r.get('index') == 64 and r['kind'] in ('terminal', 'admitted'): r['label'] = 'changed'
        elif damage == 'migration': rows.insert(1, dict(kind='relocation', step=4, arm='incident'))
        elif damage == 'cursor': rows[0]['sample_index'] = 0
        else: rows.insert(-1, copy.deepcopy(next(r for r in rows if r['kind'] == 'admitted')))
    case.edit('run/events.jsonl', change, jsonl=True)
    assert case.verify()['score'] is False


@pytest.mark.parametrize('damage', ['reset', 'rng', 'scheduler', 'missing_update', 'nonfinite',
                                  'unchanged', 'counter', 'final_delta', 'final_model', 'duplicate_load',
                                  'missing_pid', 'boolean_rank', 'final_reset', 'payload', 'ordering'])
def test_restore_and_real_update_evidence_is_required(case, damage):
    def change(rows):
        normal = next(r for r in rows if r['event'] == 'optimizer_step')
        if damage == 'reset': rows[0]['snapshot']['optimizer_steps'] = [0]
        elif damage in ('rng', 'scheduler'): rows[0][damage + '_exact'] = False
        elif damage == 'missing_update': rows.remove(normal)
        elif damage == 'nonfinite': normal['grad_norm_finite'] = False
        elif damage == 'unchanged': normal.update(sampled_changed=False, changed_tensors=0, max_delta=0)
        elif damage == 'counter': next(r for r in rows if r['event'] == 'restore_optimizer_step')['optimizer_steps'] = [1]
        elif damage == 'final_delta': rows[-1]['step_delta'] = 3
        elif damage == 'final_model': rows[-1]['snapshot']['model_sha256'] = rows[0]['snapshot']['model_sha256']
        elif damage == 'missing_pid':
            for r in rows: r.pop('pid')
        elif damage == 'boolean_rank':
            for r in rows: r['rank'] = False
        elif damage == 'final_reset': rows[-1]['snapshot']['optimizer_steps'] = [4, 4]
        elif damage == 'payload': rows[0]['payload_exact'] = False
        elif damage == 'ordering': rows[1], rows[2] = rows[2], rows[1]
        else: rows.insert(1, copy.deepcopy(rows[0]))
    case.edit('run/optimizer_evidence/actor.jsonl', change, jsonl=True)
    assert case.verify()['score'] is False


@pytest.mark.parametrize('key,value', [('start_rollout_id', 4), ('finetune', True), ('no_load_optim', True),
    ('no_load_rng', True), ('seed', 1), ('num_rollout', 4), ('lr', 1e-3), ('lr', None),
    ('dynamic_sampling_filter_path', 'reward_filter'), ('new_training_option', True),
    ('no_load_optim', 0), ('rollout_batch_size', 8.0)])
def test_config_requires_native_resume_and_unchanged_training_parameters(case, key, value):
    case.edit('run/effective_config.json', lambda c: c.update({key: value}))
    assert case.verify()['score'] is False


@pytest.mark.parametrize('damage', ['empty_inventory', 'missing_shard', 'bad_hash', 'empty_file', 'tracker',
                                  'scene_change', 'scene_extra', 'loader_change', 'input_schedule', 'failed_result'])
def test_inputs_and_checkpoint_bytes_are_required(case, damage):
    target = case.home / 'run/actor/iter_0000007/__0_0.distcp'
    if damage == 'empty_inventory': case.edit('output_checkpoints.json', lambda c: c['actor'].update(referenced_files={}))
    elif damage == 'missing_shard': target.unlink()
    elif damage == 'bad_hash': target.write_bytes(b'changed')
    elif damage == 'empty_file': target.write_bytes(b'')
    elif damage == 'tracker': (case.home / 'run/actor/latest_checkpointed_iteration.txt').write_text('3\n')
    elif damage == 'scene_change': (case.scene / 'checkpoints/actor/iter_0000003/common.pt').write_bytes(b'changed')
    elif damage == 'scene_extra': (case.scene / 'extra').write_bytes(b'changed')
    elif damage == 'loader_change': (case.home / 'submitted/loader.py').write_text('# changed\n')
    elif damage == 'input_schedule': case.edit('INPUTS.json', lambda c: c['expected_groups']['4'].reverse())
    else: case.edit('RESULT.json', lambda c: c.update(returncode=1))
    assert case.verify()['score'] is False


@pytest.mark.parametrize('damage', ['missing', 'initial', 'identity', 'label', 'count', 'repr'])
def test_only_final_fixed_structured_evaluation_is_accepted(case, damage):
    name = 'run/eval_after_step_0000007.json'
    if damage == 'missing': (case.home / name).unlink()
    elif damage == 'initial': write(case.home / 'run/eval_before_train.json', {})
    else:
        def change(value):
            samples = value['fixed']['samples']
            if damage == 'identity': samples[0]['metadata']['source_line'] = 999
            elif damage == 'label': samples[0]['label'] = 'wrong'
            elif damage == 'count': samples.pop()
            else: samples[0] = 'Sample(implicit repr is not structured evidence)'
        case.edit(name, change)
    assert case.verify()['score'] is False


def test_cli_emits_strict_json_and_no_training_imports(case):
    script = TRUSTED_BUNDLE / 'verify_resume.py'
    assert script.is_file(), 'verify_resume implementation is missing'
    proc = subprocess.run([sys.executable, '-I', '-B', str(script), '--scene', str(case.scene),
                           '--home', str(case.home)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['score'] is True


def test_archived_mode_reports_absent_bytes_but_still_rejects_changed_bytes(case):
    target = case.home / 'run/actor/iter_0000007/__0_0.distcp'
    target.unlink()
    assert case.verify()['score'] is False
    result = case.verify(fresh=False)
    assert result['score'] is True, result
    assert result['checkpoint_payloads_rehashed'] == {'actor': False, 'critic': True}
    target.write_bytes(b'SYNTHETIC changed archive')
    assert case.verify(fresh=False)['score'] is False


def test_scene_is_rehashed_after_the_audit(case, monkeypatch):
    verifier = module('verify_resume')
    evaluate = verifier._evaluation
    def change_scene(*args):
        result = evaluate(*args)
        (case.scene / 'checkpoints/critic/iter_0000003/common.pt').write_bytes(b'SYNTHETIC changed during audit')
        return result
    monkeypatch.setattr(verifier, '_evaluation', change_scene)
    result = verifier.verify_resume(case.scene, case.home)
    assert result['score'] is False
    assert 'inventory' in result['errors'][0].lower()


def test_driver_dataclass_repr_is_compared_without_source_imports(case):
    # Seal an explicit fixture of the parser's native default=str config field.
    config_path = case.scene / 'run/effective_config.json'
    config = json.loads(config_path.read_text())
    old_path = config['eval_prompt_data'][1]
    rendered = f"EvalDatasetConfig(name='fixed', path='{old_path}', n_samples_per_eval_prompt=1)"
    case.source_config({'eval_datasets': [rendered]})
    new_repr = rendered.replace(old_path, case.config['eval_prompt_data'][1])
    case.edit('run/effective_config.json', lambda c: c.update(eval_datasets=[new_repr]))
    result = case.verify()
    assert result['score'] is True, result
    case.edit('run/effective_config.json', lambda c: c.update(eval_datasets=[new_repr.replace('prompt=1', 'prompt=2')]))
    assert case.verify()['score'] is False


def test_deleted_scene_ids_and_failed_cli_return_explicit_json(case):
    path = case.scene / 'dataset/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['rows'].pop()
    write(path, manifest)
    proc = subprocess.run([sys.executable, '-I', '-B', str(TRUSTED_BUNDLE / 'verify_resume.py'), '--scene', str(case.scene),
                           '--home', str(case.home)], capture_output=True, text=True)
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result['score'] is False and result['errors']


def test_submitted_loader_is_hashed_without_execution(case):
    loader = case.home / 'submitted/loader.py'
    loader.write_text("raise AssertionError('SYNTHETIC verifier must never execute loader')\n")
    case.edit('INPUTS.json', lambda c: c.update(loader_sha256=digest(loader)))
    result = case.verify()
    assert result['score'] is True, result


def test_frozen_bundle_has_no_repository_layout_dependency(case, tmp_path):
    bundle = tmp_path / 'standalone/bundle/incident'
    bundle.mkdir(parents=True)
    for name in ('verify_resume.py', 'scene.py', 'accounting.py'):
        if (HERE / name).exists():
            shutil.copyfile(HERE / name, bundle / name)
    proc = subprocess.run([sys.executable, '-I', '-B', str(bundle / 'verify_resume.py'),
                           '--scene', str(case.scene), '--home', str(case.home)],
                          cwd=bundle, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['score'] is True


def test_transient_resolver_subprocess_failures_reconcile_retry_counts(case, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    loader = case.home / 'submitted/loader.py'
    count = case.home / 'SYNTHETIC-resolver-call-count'
    loader.write_text('from pathlib import Path\n'
        'def resolve(root, record):\n'
        f'    count = Path({str(count)!r})\n'
        '    calls = int(count.read_text()) if count.exists() else 0\n'
        '    count.write_text(str(calls + 1))\n'
        "    print('SYNTHETIC ordinary resolver log')\n"
        '    if calls < 2:\n'
        "        raise FileNotFoundError(2, 'SYNTHETIC transient lookup failure', record['path'])\n"
        "    return record['path'].replace('live/hard/', 'archive/hard/')\n")
    monkeypatch.setenv('RELOCATION_LOADER', str(loader))
    monkeypatch.setenv('RELOCATION_OUTPUT', str(case.home / 'run'))
    store = module('store').SceneStore(case.scene / 'dataset')
    row = next(r for r in case.rows if r['id'] == case.groups['4'][0])
    before = inventory(case.scene)
    assert store.read(row, sample_index=64, step=4)['id'] == row['id']
    assert count.read_text() == '3'
    events = list(map(json.loads, (case.home / 'run/events.jsonl').read_text().splitlines()))
    by_id = {r['id']: r for r in case.rows}
    for event in events:
        if event['kind'] == 'read' and event['index'] != 64:
            assert store.read(by_id[event['id']], sample_index=event['index'], step=event['step'])['id'] == event['id']
    assert count.read_text() == '66'
    case.edit('INPUTS.json', lambda c: c.update(loader_sha256=digest(loader)))
    result = case.verify()
    assert result['score'] is True, result
    assert (result['steps'][0]['read_attempts'], result['steps'][0]['abort_count'],
            result['steps'][0]['admitted_count']) == (16, 0, 16)
    trace = result['read_attempt_trace']
    assert (trace['attempts'], trace['missing'], trace['final_reads']) == (66, 2, 64)
    assert trace['steps']['4'] == dict(attempts=18, missing=2, final_reads=16)
    assert inventory(case.scene) == before


def test_wrapper_resource_import_readonly_scene_and_integer_score_mapping(case, tmp_path):
    resources = tmp_path / 'verifier/resources/incident_engine'
    resources.mkdir(parents=True)
    for name in ('verify_resume.py', 'accounting.py', 'scene.py'):
        shutil.copyfile(HERE / name, resources / name)
    spec = importlib.util.spec_from_file_location('_hidden_verify_resume', resources / 'verify_resume.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'incident').symlink_to(case.scene, target_is_directory=True)
    before = inventory(case.scene)
    permissions = {p: p.stat().st_mode & 0o7777 for p in [case.scene, *case.scene.rglob('*')]}
    try:
        for path in permissions:
            path.chmod(0o555 if path.is_dir() else 0o444)
        proof = engine.verify_resume(workspace / 'incident', case.home)
        assert proof['score'] is True, proof
        score = 1 if proof['score'] is True else 0
        assert type(score) is int and score == 1
        case.edit('RESULT.json', lambda result: result.update(returncode=1))
        proof = engine.verify_resume(workspace / 'incident', case.home)
        assert proof['score'] is False
        score = 1 if proof['score'] is True else 0
        assert type(score) is int and score == 0
        assert inventory(case.scene) == before
    finally:
        for path, mode in permissions.items():
            path.chmod(mode)


def test_plain_python_cli_no_fresh_is_explicit_and_keeps_boolean_proof(case):
    (case.home / 'run/critic/iter_0000007/__0_0.distcp').unlink()
    command = [sys.executable, '-B', str(TRUSTED_BUNDLE / 'verify_resume.py'),
               '--scene', str(case.scene), '--home', str(case.home)]
    for extra, accepted in (([], False), (['--no-fresh'], True)):
        proc = subprocess.run(command + extra, capture_output=True, text=True)
        assert proc.returncode == (0 if accepted else 1), proc.stderr
        proof = json.loads(proc.stdout)
        assert proof['score'] is accepted
        if accepted:
            assert proof['fresh'] is False
            assert proof['checkpoint_payloads_rehashed']['critic'] is False


def test_known_runtime_io_changes_preserve_training_comparison(case):
    def change(config):
        config.update(tensorboard_dir='/synthetic/new-runtime/tensorboard',
                      save_debug_rollout_data='/synthetic/new-runtime/rollout_{rollout_id}.pt')
        config['train_env_vars'].update(TENSORBOARD_DIR='/synthetic/new-runtime/tensorboard')
    case.edit('run/effective_config.json', change)
    result = case.verify()
    assert result['score'] is True, result


@pytest.mark.parametrize('field,original,actual', [
    ('gamma', 1, 1.0), ('eval_temperature', 0.0, 0),
    ('apply_chat_template_kwargs', {'enable_thinking': True}, {'enable_thinking': 1}),
    ('apply_chat_template_kwargs', {'values': [True]}, {'values': [1]}),
])
def test_config_numeric_json_equivalence_does_not_equate_booleans(case, field, original, actual):
    case.source_config({field: original})
    case.edit('run/effective_config.json', lambda config: config.update({field: actual}))
    result = case.verify()
    assert result['score'] is (field != 'apply_chat_template_kwargs'), result


@pytest.mark.parametrize('flag', ['no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng'])
@pytest.mark.parametrize('original,actual', [(None, None), (None, False), (False, None), (False, False)])
def test_native_checkpoint_flags_accept_only_documented_disabled_representations(case, flag, original, actual):
    # Megatron arguments.py:2258-2265 uses default=None for these four flags.
    case.source_config({flag: original})
    case.edit('run/effective_config.json', lambda config: config.update({flag: actual}))
    result = case.verify()
    assert result['score'] is True, result


@pytest.mark.parametrize('value', [True, 'false', 'None', '', 0, 1, [], {}])
def test_native_checkpoint_flag_never_coerces_enabled_or_arbitrary_values(case, value):
    case.edit('run/effective_config.json', lambda config: config.update(no_save_rng=value))
    assert case.verify()['score'] is False


@pytest.mark.parametrize('flag', ['finetune', 'debug_disable_optimizer', 'partial_rollout',
                                 'use_checkpoint_opt_param_scheduler', 'skip_eval_before_train'])
def test_native_none_normalization_does_not_extend_to_other_booleans(case, flag):
    case.edit('run/effective_config.json', lambda config: config.update({flag: None}))
    assert case.verify()['score'] is False


@pytest.mark.parametrize('shape', ['native_list', 'string', 'empty', 'multiple', 'wrong_path', 'null'])
def test_native_prompt_data_requires_exact_one_element_list(case, shape):
    path = str(case.scene / 'dataset/eval.jsonl')
    values = dict(native_list=[path], string=path, empty=[], multiple=[path, path],
                  wrong_path=[str(case.scene / 'dataset/manifest.json')], null=None)
    case.edit('run/effective_config.json', lambda config: config.update(prompt_data=values[shape]))
    result = case.verify()
    assert result['score'] is (shape == 'native_list'), result


def test_read_attempt_trace_is_optional_but_complete_when_present(case):
    result = case.verify()
    assert result['score'] is True, result
    assert result['read_attempt_trace'] == {'present': False}
    case.trace(transient=True)
    result = case.verify()
    assert result['score'] is True, result
    assert result['read_attempt_trace']['steps']['4'] == dict(attempts=18, missing=2, final_reads=16)
    assert result['steps'][0]['read_attempts'] == 16


@pytest.mark.parametrize('damage', ['sequence_start', 'sequence_skip', 'duplicate', 'too_many', 'id',
    'step', 'index', 'hash', 'missing_sample', 'extra_sample', 'last_outcome', 'ok_then_retry',
    'resolved_path', 'empty', 'missing_claims_hash', 'boolean_attempt'])
def test_read_attempt_trace_rejects_inconsistent_or_incomplete_attempts(case, damage):
    case.trace(transient=True)
    def change(rows):
        if damage == 'sequence_start': rows[0]['attempt'] = 2
        elif damage == 'sequence_skip': rows[1]['attempt'] = 3
        elif damage == 'duplicate': rows.insert(1, copy.deepcopy(rows[0]))
        elif damage == 'too_many':
            rows[2].update(outcome='missing', path=None, sha256=None)
            rows.insert(3, dict(rows[3], index=64, id=rows[0]['id'], attempt=4))
        elif damage in ('id', 'step', 'index'):
            rows[0][damage] = {'id': case.rows[0]['id'], 'step': 5, 'index': 65}[damage]
        elif damage == 'hash': rows[2]['sha256'] = 'f' * 64
        elif damage == 'missing_sample': rows.pop()
        elif damage == 'extra_sample': rows.append(dict(rows[-1], index=128))
        elif damage == 'last_outcome': rows[2].update(outcome='missing', path=None, sha256=None)
        elif damage == 'ok_then_retry': rows.insert(3, dict(rows[0], attempt=4))
        elif damage == 'resolved_path': rows[2]['path'] = case.rows[0]['path']
        elif damage == 'empty': rows.clear()
        elif damage == 'missing_claims_hash': rows[0]['sha256'] = rows[2]['sha256']
        else: rows[0]['attempt'] = True
    case.edit('run/read_attempts.jsonl', change, jsonl=True)
    assert case.verify()['score'] is False


def test_failed_final_read_requires_all_three_attempts(case):
    case.events(retry=True)
    case.trace()
    result = case.verify()
    assert result['score'] is True, result
    case.edit('run/read_attempts.jsonl', lambda rows: rows.pop(2), jsonl=True)
    assert case.verify()['score'] is False


def test_ref_load_can_use_sealed_actor_checkpoint_without_original_native_base(case):
    case.edit('run/effective_config.json', lambda config: config.update(ref_load=str(case.scene / 'checkpoints/actor')))
    result = case.verify()
    assert result['score'] is True, result


def test_relocated_model_assets_require_matching_sealed_byte_inventory(case):
    root, pins = case.model_assets()
    case.source_inputs({'model_files': pins})
    result = case.verify()
    assert result['score'] is True, result
    assert result['model_assets'] == dict(relocated=True, pins='scene', files_rehashed=7)
    (root / 'model.safetensors').write_bytes(b'SYNTHETIC changed model')
    assert case.verify()['score'] is False


def test_relocated_assets_can_use_scene_bound_sibling_snapshot(case, monkeypatch, tmp_path):
    _, pins = case.model_assets()
    verifier = module('verify_resume')
    trusted = tmp_path / 'trusted_bundle'
    trusted.mkdir()
    monkeypatch.setattr(verifier, 'HERE', trusted)
    snapshot = dict(schema_version=1, scene_sha256=digest(case.scene / 'SCENE.json'),
                    model_revision='b' * 40, model_files=pins)
    write(trusted / 'snapshot.json', snapshot)
    result = verifier.verify_resume(case.scene, case.home)
    assert result['score'] is True, result
    assert result['model_assets'] == dict(relocated=True, pins='snapshot', files_rehashed=7)
    snapshot['scene_sha256'] = '0' * 64
    write(trusted / 'snapshot.json', snapshot)
    result = verifier.verify_resume(case.scene, case.home)
    assert result['score'] is False
    assert 'scene' in result['errors'][0].lower()


@pytest.mark.parametrize('damage', ['no_pins', 'missing_file', 'extra_file', 'incomplete_pins',
                                  'tokenizer_path', 'ref_path', 'fallback'])
def test_model_relocation_keeps_identity_and_restore_constraints(case, damage):
    root, pins = case.model_assets()
    if damage == 'incomplete_pins': pins.pop('tokenizer.json')
    if damage != 'no_pins': case.source_inputs({'model_files': pins})
    if damage == 'missing_file': (root / 'config.json').unlink()
    elif damage == 'extra_file': (root / 'unexpected.json').write_text('{}')
    elif damage == 'tokenizer_path':
        case.edit('run/effective_config.json', lambda config: config.update(tokenizer_model='/synthetic/other-tokenizer'))
    elif damage == 'ref_path':
        case.edit('run/effective_config.json', lambda config: config.update(ref_load=str(case.scene / 'checkpoints/critic')))
    elif damage == 'fallback':
        case.edit('run/effective_config.json', lambda config: config.update(pretrained_checkpoint=str(root)))
    assert case.verify()['score'] is False


def test_relocated_snapshot_inventory_cannot_disagree_with_sealed_source(case, monkeypatch, tmp_path):
    _, pins = case.model_assets()
    case.source_inputs({'model_files': pins})
    verifier = module('verify_resume')
    monkeypatch.setattr(verifier, 'HERE', tmp_path)
    write(tmp_path / 'snapshot.json', dict(scene_sha256=digest(case.scene / 'SCENE.json'),
          model_files={**pins, 'model.safetensors': 'f' * 64}))
    assert verifier.verify_resume(case.scene, case.home)['score'] is False


def test_task_wrapper_reads_parent_snapshot_pins_and_converts_boolean_score(case, tmp_path):
    """Run the real packaging wrapper only on isolated, explicitly synthetic evidence."""
    model, pins = case.model_assets()
    verifier_root = tmp_path / 'copied_verifier'
    resources = verifier_root / 'resources'
    engine = resources / 'incident_audit'
    engine.mkdir(parents=True)
    for name in ('verify_resume.py', 'accounting.py', 'scene.py'):
        shutil.copyfile(HERE / name, engine / name)
    wrapper = verifier_root / 'tests/grade_submission.py'
    wrapper.parent.mkdir()
    source = HERE.parents[2] / 'tasks/data/justrl2_data_relocation/evaluations/solution/verifier/tests/grade_submission.py'
    shutil.copyfile(source, wrapper)
    write(resources / 'snapshot.json', dict(scene_sha256=digest(case.scene / 'SCENE.json'),
          model_revision='b' * 40, model_files=pins))
    prepared, results = tmp_path / 'prepared_root', tmp_path / 'wrapper_results'
    prepared.mkdir()
    (prepared / 'incident').symlink_to(case.scene, target_is_directory=True)
    (results / 'experiment_logs').mkdir(parents=True)
    (results / 'experiment_logs/run-home').symlink_to(case.home, target_is_directory=True)
    env = dict(os.environ, AUTORLBENCH_PREPARED=str(prepared), AUTORLBENCH_RESULTS=str(results))
    before = inventory(case.scene)
    for accepted in (True, False):
        if not accepted:
            (model / 'model.safetensors').write_bytes(b'SYNTHETIC changed model bytes')
        process = subprocess.run([sys.executable, '-I', '-B', str(wrapper)],
                                 env=env, capture_output=True, text=True)
        assert process.returncode == (0 if accepted else 1), process.stdout + process.stderr
        result = json.loads((results / 'result.json').read_text())
        assert result['passed'] is accepted and result['metrics']['score'] is accepted
        assert type(result['score']) is int and result['score'] == int(accepted)
        if accepted:
            assert result['metrics']['model_assets']['files_rehashed'] == 7
    assert inventory(case.scene) == before


@pytest.fixture
def captured_scene_config():
    """Opt-in real prefix1430222 JSON only; no weights, training, or full-scene audit."""
    name = os.environ.get('INCIDENT_SCENE_RECEIPT')
    if not name:
        pytest.skip('Set INCIDENT_SCENE_RECEIPT to the fetched real SCENE.json for config-only checks')
    path = Path(name)
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == 'b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac'
    receipt = json.loads(original)
    assert (receipt['schema_version'], receipt['checkpoint_iteration'], receipt['resume_start']) == (1, 3, 4)
    yield path.parent, receipt
    assert path.read_bytes() == original


def resumed_native_config(receipt, scene, home, disabled=False, portable=False):
    """Project the frozen continuation CLI onto captured args; this is not a run artifact."""
    base = receipt['source']['parameters']
    config = copy.deepcopy(base)
    config.update(start_rollout_id=None, finetune=False, no_load_optim=disabled, no_load_rng=disabled,
        use_checkpoint_opt_param_scheduler=True, save_retain_interval=4,
        load=str(scene / 'checkpoints/actor'), critic_load=str(scene / 'checkpoints/critic'),
        save=str(home / 'run/actor'), critic_save=str(home / 'run/critic'),
        prompt_data=[str(scene / 'dataset/eval.jsonl')],
        eval_prompt_data=['fixed', str(scene / 'dataset/eval.jsonl')])
    if portable:
        config.update(hf_checkpoint=str(home.parent / 'prepared/assets/hf-safetensors'),
                      tokenizer_model=str(home.parent / 'prepared/assets/hf-safetensors'),
                      ref_load=str(scene / 'checkpoints/actor'))
    for key in ('data_source_path', 'rollout_function_path', 'eval_function_path', 'custom_generate_function_path'):
        config[key] = config[key].replace('relocation_rollout.', 'rollout.')
    config['eval_datasets'] = [value.replace(base['eval_prompt_data'][1], config['eval_prompt_data'][1])
                               for value in config['eval_datasets']]
    config['train_env_vars'].update(RELOCATION_DATA=str(scene / 'dataset'), RELOCATION_OUTPUT=str(home / 'run'),
        RELOCATION_ARM='incident', RELOCATION_STEPS='8', RELOCATION_AT='4',
        BENCH_RESTORE_ROOT=str(scene / 'checkpoints'), BENCH_RESTORE_ITERATION='3',
        RELOCATION_MODEL=config['hf_checkpoint'], RELOCATION_NATIVE=config['ref_load'])
    return config


@pytest.mark.parametrize('portable', [False, True])
@pytest.mark.parametrize('disabled', [False, None])
def test_real_scene_config_preserves_reference_and_portable_model_policy(captured_scene_config, tmp_path, portable, disabled):
    scene, receipt = captured_scene_config
    base, home = receipt['source']['parameters'], tmp_path / 'continuation'
    assert len(base) == 1546 and base['no_save_rng'] is None and isinstance(base['prompt_data'], list)
    config = resumed_native_config(receipt, scene, home, disabled, portable)
    verifier = module('verify_resume')
    verifier._config(config, base, scene, home)
    if portable:
        # The actual sealed receipt does not yet contain model_files. Do not
        # invent those pins from an older run or turn paths into byte evidence.
        with pytest.raises(ValueError, match='scene-bound model_files pins'):
            verifier._model_assets(scene, receipt, config)
    else:
        assert config['hf_checkpoint'] == base['hf_checkpoint'] and config['ref_load'] == base['ref_load']
        assert verifier._model_assets(scene, receipt, config) == dict(relocated=False, pins=None, files_rehashed=0)


@pytest.mark.parametrize('key,value', [('no_load_optim', True), ('num_rollout', 4), ('lr', 0.001)])
def test_real_scene_config_still_rejects_recipe_drift(captured_scene_config, tmp_path, key, value):
    scene, receipt = captured_scene_config
    home = tmp_path / 'continuation'
    config = resumed_native_config(receipt, scene, home, portable=True)
    config[key] = value
    with pytest.raises(ValueError):
        module('verify_resume')._config(config, receipt['source']['parameters'], scene, home)
