"""Explicit synthetic setup shared by the bounded review suite; no test functions.

Checkpoint byte fixtures are deliberately unloadable. Real CPU tensor fixtures
live separately in restore/fixtures.py. No historical training proof is reused.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'tasks/data/justrl2_data_relocation/environment/incident_engine'
FAULTY = REPO / 'tasks/data/justrl2_data_relocation'
REFERENCE_LOADER = FAULTY / 'evaluations/solution/oracle/reference_loader.py'
VERIFIER = FAULTY / 'evaluations/solution/verifier'
AUDIT = VERIFIER / 'resources/incident_audit'


def load(path):
    spec = importlib.util.spec_from_file_location('_review_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def module(name):
    return load(SOURCE / (name + '.py'))


def copy_auditor(root):
    root.mkdir(parents=True)
    for name in ('verify_resume.py', 'accounting.py', 'scene.py'):
        shutil.copyfile(AUDIT / name, root / name)
    return root


def clean_env(**values):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('AUTORLBENCH_', 'RELOCATION_', 'BENCH_RESTORE_'))
           and key not in ('PYTHONPATH', 'PYTHONHOME')}
    return dict(env, PYTHONDONTWRITEBYTECODE='1', **{key: str(value) for key, value in values.items()})


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
        self.bundle = copy_auditor(root / 'trusted/incident')
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
        result = load(self.bundle / 'verify_resume.py').verify_resume(self.scene, self.home, **kwargs)
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

class ReviewRun:
    """Temporary wrapper package and report; always run the real stdlib auditor."""
    def __init__(self, root, expected, case):
        self.case, self.expected = case, expected
        self.workspace, self.results, self.prepared = (root / name for name in ('workspace','results','prepared'))
        self.workspace.mkdir(parents=True)
        self.prepared.mkdir()
        (self.results / 'experiment_logs').mkdir(parents=True)
        (self.prepared / 'incident').symlink_to(case.scene, target_is_directory=True)
        (self.results / 'experiment_logs/run-home').symlink_to(case.home, target_is_directory=True)
        self.wrapper = root / 'verifier/tests/grade_submission.py'
        self.wrapper.parent.mkdir(parents=True)
        shutil.copyfile(VERIFIER / 'tests/grade_submission.py', self.wrapper)
        resources = self.wrapper.parent.parent / 'resources'
        copy_auditor(resources / 'incident_audit')
        write(resources / 'snapshot.json', {'scene_sha256': digest(case.scene / 'SCENE.json')})
        write(resources / 'expected_verdict.json', {'schema_version': 1, 'expected_verdict': expected})
        start = REFERENCE_LOADER if expected == 'healthy' else FAULTY / 'workspace/loader.py'
        self.loader = self.workspace / 'loader.py'
        shutil.copyfile(start, self.loader)
        self.initial_loader = self.loader.read_bytes()

    def submit(self, report=None, repair=False, archived=True):
        if repair:
            shutil.copyfile(REFERENCE_LOADER, self.loader)
        if report is not None:
            ((self.results if archived else self.workspace) / 'report.md').write_text(report)
        shutil.copyfile(self.loader, self.case.home / 'submitted/loader.py')
        self.case.edit('INPUTS.json', lambda value: value.update(loader_sha256=digest(self.loader)))

    def run(self):
        env = clean_env(AUTORLBENCH_RESULTS=self.results, AUTORLBENCH_PREPARED=self.prepared,
                        AUTORLBENCH_WORKSPACE=self.workspace)
        process = subprocess.run([sys.executable, '-I', '-B', str(self.wrapper)],
                                 env=env, capture_output=True, text=True)
        result = json.loads((self.results / 'result.json').read_text())
        assert process.returncode == (0 if result['passed'] else 1), process.stderr
        assert json.loads((self.results / 'metrics.json').read_text()) == result
        assert json.loads(process.stdout) == result
        return result, process.returncode
