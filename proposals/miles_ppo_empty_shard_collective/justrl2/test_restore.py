# SPDX-License-Identifier: Apache-2.0
"""Real CPU Adam/DCP tests; these are not GPU restore evidence."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def helper():
    assert (HERE / 'restore_evidence.py').is_file(), 'restore validation helper is missing'
    return importlib.import_module('restore_evidence')


def state_after_update():
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).square().sum().backward()
    opt.step()
    return model, opt


def state_dict(model, opt):
    return {'model': model.state_dict(), 'optimizer': opt.state_dict()}


def test_actual_adam_checkpoint_restore_and_next_update(tmp_path):
    h = helper()
    model, opt = state_after_update()
    saved = copy.deepcopy(state_dict(model, opt))
    before = h.fingerprint(saved)
    dcp.save(saved, checkpoint_id=tmp_path / 'checkpoint')
    expected = copy.deepcopy(saved)
    for value in expected['model'].values():
        value.zero_()
    dcp.load(expected, checkpoint_id=tmp_path / 'checkpoint')
    fresh_model = torch.nn.Linear(3, 2)
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=0.01)
    fresh_model.load_state_dict(expected['model'])
    fresh_opt.load_state_dict(expected['optimizer'])
    restored = h.compare_state(state_dict(fresh_model, fresh_opt), expected)
    assert restored == before
    assert restored['optimizer_steps'] == [1.0, 1.0]
    fresh_opt.zero_grad()
    fresh_model(torch.ones(2, 3)).square().sum().backward()
    fresh_opt.step()
    updated = h.fingerprint(state_dict(fresh_model, fresh_opt))
    assert updated['optimizer_steps'] == [2.0, 2.0]
    assert updated['model_sha256'] != before['model_sha256']
    assert updated['optimizer_sha256'] != before['optimizer_sha256']
    assert h.fingerprint(saved) == before


@pytest.mark.parametrize('damage', ['model', 'moment', 'master', 'step', 'missing', 'nan'])
def test_rejects_incorrect_or_incomplete_restore(damage):
    h = helper()
    model, opt = state_after_update()
    expected = copy.deepcopy(state_dict(model, opt))
    expected['optimizer']['master'] = torch.arange(2049, dtype=torch.float32)
    actual = copy.deepcopy(expected)
    if damage == 'model': actual['model']['weight'].view(-1)[-1] += 1
    if damage == 'moment': actual['optimizer']['state'][0]['exp_avg'].view(-1)[-1] += 1
    if damage == 'master': actual['optimizer']['master'][1023] += 1
    if damage == 'step': actual['optimizer']['state'][0]['step'] += 1
    if damage == 'missing': del actual['optimizer']['state'][0]['exp_avg_sq']
    if damage == 'nan': actual['model']['weight'][0, 0] = float('nan')
    with pytest.raises(ValueError): h.compare_state(actual, expected)


def test_fresh_adam_is_not_a_restored_optimizer():
    h = helper()
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match='Adam|step|moment'):
        h.fingerprint(state_dict(model, opt))


def test_digest_handles_bfloat16_and_is_independent_of_dict_order():
    h = helper()
    model, opt = state_after_update()
    state = state_dict(model, opt)
    state['model']['bf16'] = torch.arange(4099, dtype=torch.bfloat16)
    assert h.fingerprint(state) == h.fingerprint(dict(reversed(list(state.items()))))


def test_checked_load_rejects_swapped_adam_moments_with_identical_tensor_multiset():
    h = helper()
    model, opt = state_after_update()
    actual = state_dict(model, opt)
    expected = copy.deepcopy(actual)
    st = expected['optimizer']['state'][0]
    st['exp_avg'], st['exp_avg_sq'] = st['exp_avg_sq'], st['exp_avg']
    def generate(optim_sd_kwargs=None): return actual
    captured = dict(generate=generate, positional=(), keywords={}, load_kwargs={})
    with pytest.raises(ValueError, match='differs'):
        h.checked_load(lambda: (0, 0), captured, lambda *a, **kw: expected,
                       Path('/unused'), 0)


def test_state_difference_reports_exact_persistent_field_and_samples():
    h = helper()
    model, opt = state_after_update()
    actual = state_dict(model, opt)
    expected = copy.deepcopy(actual)
    expected['optimizer']['state'][0]['exp_avg'][0, 0] += 1
    differences = h.state_differences(actual, expected)
    assert len(differences) == 1
    assert differences[0]['path'] == ['optimizer', 'state', '0', 'exp_avg']
    assert differences[0]['kind'] == 'tensor_values'
    assert differences[0]['different_elements'] == 1
    assert differences[0]['samples'][0]['flat_index'] == 0
    assert abs(differences[0]['samples'][0]['expected'] - differences[0]['samples'][0]['actual'] - 1) < 1e-6


def restore_args(tmp_path):
    root = tmp_path / 'original'
    for role in ('actor', 'critic'):
        (root / role / 'iter_0000000').mkdir(parents=True)
        (root / role / 'latest_checkpointed_iteration.txt').write_text('0')
    return root, SimpleNamespace(load=str(root / 'actor'), critic_load=str(root / 'critic'),
        save=str(tmp_path / 'new' / 'actor'), critic_save=str(tmp_path / 'new' / 'critic'),
        finetune=False, no_load_optim=False, no_save_optim=False, debug_disable_optimizer=False,
        start_rollout_id=None, num_rollout=2, async_save=False, ckpt_step=None,
        megatron_to_hf_mode='raw', use_critic=True, debug_train_only=True,
        context_parallel_size=2, actor_num_gpus_per_node=4, critic_num_gpus_per_node=4)


@pytest.mark.parametrize('damage', ['none', 'finetune', 'no_load_optim', 'no_save_optim',
    'start', 'load', 'critic', 'overlap', 'missing', 'iterations'])
def test_restore_arguments_fail_closed(tmp_path, damage):
    h = helper()
    root, args = restore_args(tmp_path)
    if damage in ('finetune','no_load_optim','no_save_optim'): setattr(args, damage, True)
    if damage == 'start': args.start_rollout_id = 0
    if damage == 'load': args.load = str(tmp_path / 'native')
    if damage == 'critic': args.critic_load = args.load
    if damage == 'overlap': args.save = str(root / 'actor' / 'child')
    if damage == 'missing': (root / 'critic/latest_checkpointed_iteration.txt').unlink()
    if damage == 'iterations': args.num_rollout = 1
    if damage == 'none': h.validate_restore_args(args, root, 0)
    else:
        with pytest.raises(ValueError): h.validate_restore_args(args, root, 0)


def test_reconstructed_template_uses_save_state_without_mutating_load_kwargs():
    h = helper()
    calls = []
    def generate(args, model, optimizer, opt_param_scheduler, rng_state,
                 iteration=None, optim_sd_kwargs=None):
        calls.append(optim_sd_kwargs.copy())
        return {'model': {'weight': torch.ones(2)}, 'optimizer': {'test': 1}}
    kw = {'optim_sd_kwargs': {'is_loading': True, 'metadata': {'format': 'dp_reshardable'}}}
    state = h.regenerate_state(generate, (None, None, None, None, None), kw)
    assert set(state) == {'model','optimizer'}
    assert calls[0]['is_loading'] is False
    assert kw['optim_sd_kwargs']['is_loading'] is True


def test_cpu_template_has_no_aliases_to_live_tensors():
    h = helper()
    assert hasattr(h, 'cpu_template'), 'independent checkpoint read buffers are missing'
    live = {'model': {'weight': torch.arange(8.)}, 'optimizer': {'step': torch.tensor(1.)}}
    template = h.cpu_template(live)
    template['model']['weight'].fill_(99)
    template['optimizer']['step'].fill_(99)
    assert torch.equal(live['model']['weight'], torch.arange(8.))
    assert live['optimizer']['step'].item() == 1


def test_next_update_requires_exactly_one_step_and_changed_moments():
    h = helper()
    assert hasattr(h, 'validate_progress'), 'next-update validation is missing'
    model, opt = state_after_update()
    before = h.fingerprint(state_dict(model, opt))
    opt.zero_grad()
    model(torch.ones(2, 3)).square().sum().backward()
    opt.step()
    after = h.fingerprint(state_dict(model, opt))
    h.validate_progress(before, after)
    for field in ('optimizer_sha256', 'optimizer_steps'):
        bad = {**after, field: before[field]}
        with pytest.raises(ValueError): h.validate_progress(before, bad)
    with pytest.raises(ValueError):
        h.validate_progress(before, {**after, 'optimizer_steps': [3., 3.]})


def test_native_hook_compares_real_live_adam_and_rejects_noop_loader(tmp_path, monkeypatch):
    """Dependency seam mimics Megatron entry points; real Torch Adam owns state."""
    h = helper()
    assert hasattr(h, 'checked_load'), 'native restore gate is missing'
    source_model, source_opt = state_after_update()
    expected = copy.deepcopy(state_dict(source_model, source_opt))
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    def generate(*args, **kwargs):
        return state_dict(model, opt)
    captured = {'generate': generate, 'positional': (), 'keywords': {},
                'load_kwargs': {}, 'load_args': ()}
    # Separate read really uses DCP and cannot mutate the live model/optimizer.
    dcp.save(expected, checkpoint_id=tmp_path / 'ckpt')
    def reader(template, path, **kwargs):
        dcp.load(template, checkpoint_id=path)
        return template
    def load_ok():
        model.load_state_dict(expected['model']); opt.load_state_dict(expected['optimizer'])
        return (0, 0)
    # A real generate function must accept optim_sd_kwargs.
    def generate(args=None, model_arg=None, optimizer=None, opt_param_scheduler=None,
                 rng_state=None, optim_sd_kwargs=None):
        return state_dict(model, opt)
    captured['generate'] = generate
    fp = h.checked_load(load_ok, captured, reader, tmp_path / 'ckpt', 0)
    assert fp == h.fingerprint(expected)
    with torch.no_grad(): model.weight.add_(1)
    with pytest.raises(ValueError, match='differs'):
        h.checked_load(lambda: (0, 0), captured, reader, tmp_path / 'ckpt', 0)
    with pytest.raises(ValueError, match='iteration'):
        h.checked_load(lambda: (1, 0), captured, reader, tmp_path / 'ckpt', 0)


def launcher():
    assert (HERE / 'run_restore.py').is_file(), 'restore launcher is missing'
    return importlib.import_module('run_restore')


def test_instrumentation_targets_native_load_and_post_save():
    r = launcher()
    text = '''import os
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if hashes is not None:
'''
    updated = r.instrument_model(text)
    assert 'restore_load_checkpoint(args, load_checkpoint,' in updated
    assert 'restore_after_save(args, iteration)' in updated
    assert r.instrument_model(updated) == updated
    with pytest.raises(ValueError): r.instrument_model('unsupported source')


def test_launcher_never_overlays_original_prepared_tree(tmp_path):
    r = launcher()
    prepared = tmp_path / 'prepared'
    source = prepared / 'pair/healthy'
    (source / 'benchmark').mkdir(parents=True)
    (source / 'miles/backends/megatron_utils').mkdir(parents=True)
    (source / 'miles/backends/megatron_utils/model.py').write_text('    iteration, _ = load_checkpoint(\n    if hashes is not None:\n')
    (source / 'keep.txt').write_text('original')
    framework = tmp_path / 'framework'
    framework.mkdir()
    (source / 'Megatron-LM').symlink_to(framework, target_is_directory=True)
    workspace = r.copy_workspace(prepared, tmp_path / 'new')
    assert (workspace / 'keep.txt').read_text() == 'original'
    assert (workspace / 'Megatron-LM').resolve() == framework
    assert not (source / 'benchmark/restore_evidence.py').exists()
    assert (workspace / 'benchmark/restore_evidence.py').is_file()
    with pytest.raises(FileExistsError): r.copy_workspace(prepared, tmp_path / 'new')


def test_restore_shell_uses_separate_load_roots_and_infers_start(tmp_path):
    import os, subprocess
    r = launcher()
    root = tmp_path / 'workspace'
    (root / 'benchmark').mkdir(parents=True)
    (root / 'justrl2/model_args').mkdir(parents=True)
    (root / 'justrl2/model_args/minicpm5-2b.sh').write_text('MODEL_ARGS=()\n')
    (root / 'benchmark/train.sh').write_text((HERE / 'train.sh').read_text())
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'python3').write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    (binary / 'python3').chmod(0o755)
    env = {**os.environ, 'PATH': str(binary) + ':' + os.environ['PATH'],
        'BENCH_MODEL': '/model', 'BENCH_NATIVE_MODEL': '/native', 'BENCH_REPLAY': '/replay.pt',
        'BENCH_RUN_DIR': str(tmp_path / 'output'), 'BENCH_RESTORE_ROOT': '/original',
        'BENCH_RESTORE_ITERATION': '0'}
    result = subprocess.run(['bash', str(root / 'benchmark/train.sh')], env=env,
                            text=True, capture_output=True, check=True).stdout.splitlines()
    assert result[result.index('--load')+1] == '/original/actor'
    assert result[result.index('--critic-load')+1] == '/original/critic'
    assert result[result.index('--num-rollout')+1] == '2'
    assert '--start-rollout-id' not in result
    env.pop('BENCH_RESTORE_ROOT')
    result = subprocess.run(['bash', str(root / 'benchmark/train.sh')], env=env,
                            text=True, capture_output=True, check=True).stdout.splitlines()
    assert result[result.index('--num-rollout')+1] == '1'
    assert result[result.index('--start-rollout-id')+1] == '0'


def evidence_rows():
    model, opt = state_after_update()
    h = helper()
    before = h.fingerprint(state_dict(model, opt))
    opt.zero_grad(); model(torch.ones(2, 3)).square().sum().backward(); opt.step()
    after = h.fingerprint(state_dict(model, opt))
    rows = []
    for role in ('actor','critic'):
        for rank in range(4):
            identity = dict(role=role, rank=rank, pid=100+rank, cp_size=2, dp_size=2,
                cp_rank=rank%2, dp_rank=rank//2, cp_members=[rank//2*2, rank//2*2+1],
                dp_members=[rank%2, rank%2+2])
            rows.extend([
                dict(identity, event='restore_loaded', loaded_iteration=0, next_rollout_id=1,
                    payload_exact=True, load_root='/original/'+role, snapshot=before),
                dict(identity, event='optimizer_step', rollout_id=1, step_id=0,
                    update_successful=True, grad_norm_finite=True, sampled_finite=True, sampled_changed=True),
                dict(identity, event='restore_saved', saved_iteration=1, payload_exact=True,
                    save_root='/new/'+role, snapshot=after)])
    return rows


@pytest.mark.parametrize('damage', ['none','missing_rank','duplicate','no_step','wrong_step',
    'wrong_iteration','no_moments','no_change','wrong_group','order','path','payload'])
def test_restore_checker_requires_each_rank_and_one_actual_update(damage):
    r = launcher()
    assert hasattr(r, 'verify_events'), 'restore evidence checker is missing'
    rows = evidence_rows()
    if damage == 'missing_rank': rows = [x for x in rows if not(x['role']=='critic' and x['rank']==3)]
    if damage == 'duplicate': rows.append(rows[0].copy())
    if damage == 'no_step': rows[1]['update_successful'] = False
    if damage == 'wrong_step': rows[1]['rollout_id'] = 0
    if damage == 'wrong_iteration': rows[2]['saved_iteration'] = 0
    if damage == 'no_moments': rows[0]['snapshot'] = {**rows[0]['snapshot'], 'optimizer_steps': []}
    if damage == 'no_change':
        for row in rows:
            if row['event']=='optimizer_step': row['sampled_changed'] = False
    if damage == 'wrong_group': rows[0]['dp_members'] = [0,1]
    if damage == 'order': rows[0], rows[2] = rows[2], rows[0]
    if damage == 'path': rows[0]['load_root'] = '/native'
    if damage == 'payload': rows[0]['payload_exact'] = False
    if damage == 'none': assert r.verify_events(rows, Path('/original'), Path('/new'), 0)['ranks_verified'] == 8
    else:
        with pytest.raises(ValueError): r.verify_events(rows, Path('/original'), Path('/new'), 0)


def test_worker_hook_preserves_native_return_and_keeps_args_serializable(tmp_path, monkeypatch):
    from types import ModuleType
    import pickle
    h = helper()
    model, opt = state_after_update()
    expected = copy.deepcopy(state_dict(model,opt))
    root = tmp_path/'original/actor'
    root.mkdir(parents=True)
    (root/'latest_checkpointed_iteration.txt').write_text('0')
    checkpointing = ModuleType('megatron.training.checkpointing')
    dc = ModuleType('megatron.core.dist_checkpointing')
    def generate(args, model, optimizer, scheduler, rng, optim_sd_kwargs=None):
        return state_dict(model, optimizer)
    def read(sharded_state_dict, checkpoint_dir, sharded_strategy=None, validate_access_integrity=True, **kw):
        return copy.deepcopy(expected)
    checkpointing.generate_state_dict, dc.load = generate, read
    for name in ('megatron','megatron.training','megatron.core','benchmark'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    monkeypatch.setitem(sys.modules, checkpointing.__name__, checkpointing)
    monkeypatch.setitem(sys.modules, dc.__name__, dc)
    evidence = ModuleType('benchmark.runtime_evidence')
    records=[]
    evidence.emit=lambda args,event,**fields:records.append({'event':event,**fields})
    monkeypatch.setitem(sys.modules,evidence.__name__,evidence)
    monkeypatch.setenv('BENCH_RESTORE_ROOT',str(root.parent))
    monkeypatch.setenv('BENCH_RESTORE_ITERATION','0')
    args=SimpleNamespace(load=str(root),bench_role='actor',finetune=False,no_load_optim=False,no_save_optim=False)
    def native(model,optimizer,scheduler,**kw):
        template=checkpointing.generate_state_dict(args,model,optimizer,scheduler,None,
                optim_sd_kwargs={'is_loading':True})
        dc.load(template,root/'iter_0000000',validate_access_integrity=False)
        return (0, 12345)
    result=h.restore_load_checkpoint(args,native,model,opt,None)
    assert result == (0,12345)
    pickle.dumps(args)
    assert records[0]['payload_exact'] is True
    assert checkpointing.generate_state_dict is generate and dc.load is read


class LocalNonpersistentObject:
    """Megatron wrapper seam; real tensors exercise padding preservation."""
    __module__ = 'megatron.core.dist_checkpointing.mapping'

    def __init__(self, obj):
        self.obj = obj

    def unwrap(self):
        return self.obj


def test_nonpersistent_padding_template_preserves_values_without_aliasing():
    h = helper()
    padding = torch.tensor([3., 7.])
    template = h.cpu_template({'padding': LocalNonpersistentObject(padding)})
    torch.testing.assert_close(template['padding'].unwrap(), padding)
    template['padding'].unwrap()[0] = 99
    assert padding[0].item() == 3


def test_nonpersistent_padding_is_excluded_but_persistent_padding_is_checked():
    h = helper()
    model, opt = state_after_update()
    state = state_dict(model, opt)
    expected = h.fingerprint(state)
    state['optimizer']['padding'] = LocalNonpersistentObject(torch.tensor([float('nan')]))
    assert h.fingerprint(state) == expected
    state['optimizer']['padding'] = torch.tensor([float('nan')])
    with pytest.raises(ValueError, match='Nonfinite'):
        h.fingerprint(state)


def test_checked_load_excludes_unwrapped_nonpersistent_padding_only():
    h = helper()
    model, opt = state_after_update()
    actual = state_dict(model, opt)
    expected = copy.deepcopy(actual)
    expected_hash = h.fingerprint(actual)
    actual['optimizer']['padding'] = LocalNonpersistentObject(torch.tensor([float('nan')]))
    expected['optimizer']['padding'] = torch.tensor([float('nan')])
    def generate(optim_sd_kwargs=None):
        return actual
    captured = dict(generate=generate, positional=(), keywords={}, load_kwargs={})
    def reader(template, selected):
        return expected
    assert h.checked_load(lambda: (0, 0), captured, reader, Path('/unused'), 0) == expected_hash
    expected['optimizer']['state'][0]['exp_avg'][0, 0] += 1
    with pytest.raises(ValueError, match='differs'):
        h.checked_load(lambda: (0, 0), captured, reader, Path('/unused'), 0)

class ShardedTensor:
    __module__ = 'megatron.core.dist_checkpointing.mapping'
    def __init__(self, data):
        self.data = data
        self.key = 'optimizer.distributed.test_bucket'


def test_native_dp_reshardable_alignment_padding_is_not_optimizer_state():
    h = helper()
    model, opt = state_after_update()
    actual = state_dict(model, opt)
    # Same representation as pinned sharded_param_state_dp_reshardable:
    # padding tensors are allocated with torch.empty and dropped by native load.
    actual['optimizer']['param_state'] = {'bucket': [{
        'padding': LocalNonpersistentObject(True),
        'param': ShardedTensor(torch.tensor([17.])),
        'exp_avg': ShardedTensor(torch.tensor([23.])),
        'exp_avg_sq': ShardedTensor(torch.tensor([31.])),
    }]}
    expected = copy.deepcopy(actual)
    bucket = expected['optimizer']['param_state']['bucket'][0]
    bucket['padding'] = True
    for name in ('param', 'exp_avg', 'exp_avg_sq'):
        bucket[name] = torch.tensor([99.])
    def generate(optim_sd_kwargs=None): return actual
    cap = dict(generate=generate,positional=(),keywords={},load_kwargs={})
    restored = h.checked_load(lambda: (0, 0),cap,lambda *a, **kw: expected,Path('/unused'),0)
    assert restored['optimizer_steps'] == [1.,1.]
    assert restored['ignored_storage_padding_elements'] == 3
    assert restored['ignored_storage_padding_tensors'] == 3
    # Damage outside explicit native padding must still fail.
    expected['optimizer']['state'][0]['exp_avg'][0,0] += 1
    with pytest.raises(ValueError,match='differs'):
        h.checked_load(lambda: (0, 0),cap,lambda *a, **kw: expected,Path('/unused'),0)


def test_plain_padding_marker_never_excludes_persistent_tensors():
    h = helper()
    model,opt = state_after_update()
    state=state_dict(model,opt)
    state['optimizer']['param_state'] = {'bucket':[{'padding': True,'param':torch.ones(2)}]}
    assert h.fingerprint(state)['optimizer_elements'] > h.fingerprint(state_dict(model,opt))['optimizer_elements']
