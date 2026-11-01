# SPDX-License-Identifier: Apache-2.0
"""Isolated synthetic CPU Torch/DCP fixtures; not native GPU restore evidence.

Fingerprint regression cases adapted from the Apache-2.0 JustRL2 restore tests.
License: ../runtime_support/LICENSE.
"""
import copy
import importlib.util
import json
import pickle
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

HERE = Path(__file__).resolve().parent
_ADAPTER = None


def helper():
    global _ADAPTER
    path = HERE / 'restore_evidence.py'
    assert path.is_file(), 'incident restore adapter is missing'
    if _ADAPTER is None:
        spec = importlib.util.spec_from_file_location('incident_restore_evidence', path)
        _ADAPTER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ADAPTER)
    return _ADAPTER


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


def scene_args(tmp_path):
    root = tmp_path / 'original'
    for role in ('actor', 'critic'):
        (root / role / 'iter_0000003').mkdir(parents=True)
        (root / role / 'latest_checkpointed_iteration.txt').write_text('3')
    args = SimpleNamespace(load=str(root / 'actor'), critic_load=str(root / 'critic'),
        save=str(tmp_path / 'new/actor'), critic_save=str(tmp_path / 'new/critic'),
        bench_role='actor', start_rollout_id=None, num_rollout=8,
        ckpt_step=None, ckpt_format='torch_dist', megatron_to_hf_mode='raw',
        use_critic=True, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_model_parallel_size=1, data_parallel_size=1,
        actor_num_nodes=1, critic_num_nodes=1, actor_num_gpus_per_node=1,
        critic_num_gpus_per_node=1, rollout_num_gpus=1, rollout_num_gpus_per_engine=1,
        num_gpus_per_node=3, global_batch_size=16,
        finetune=False, no_load_optim=False, no_save_optim=False,
        no_load_rng=False, no_save_rng=False, async_save=False,
        override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=True)
    return root, args


@pytest.mark.parametrize('damage', ['none', 'finetune', 'no_load_optim', 'no_save_optim',
    'no_load_rng', 'no_save_rng', 'debug_disable_optimizer', 'override_opt_param_scheduler',
    'async_save', 'start', 'budget', 'iteration', 'tracker', 'missing', 'load', 'overlap',
    'save_overlap', 'topology', 'pretrained_checkpoint'])
def test_incident_contract(tmp_path, damage):
    h = helper()
    root, args = scene_args(tmp_path)
    iteration = 3
    if damage in ('finetune', 'no_load_optim', 'no_save_optim', 'no_load_rng', 'no_save_rng',
                  'debug_disable_optimizer', 'override_opt_param_scheduler', 'async_save'):
        setattr(args, damage, True)
    if damage == 'start': args.start_rollout_id = 4
    if damage == 'budget': args.num_rollout = 5
    if damage == 'iteration': iteration = 0
    if damage == 'tracker': (root / 'critic/latest_checkpointed_iteration.txt').write_text('release')
    if damage == 'missing': (root / 'critic/iter_0000003').rmdir()
    if damage == 'load': args.critic_load = args.load
    if damage == 'overlap': args.save = str(root.parent)
    if damage == 'save_overlap': args.critic_save = args.save
    if damage == 'topology': args.context_parallel_size = 2
    if damage == 'pretrained_checkpoint': args.pretrained_checkpoint = '/fallback'
    if damage == 'none': h.validate_restore_args(args, root, iteration)
    else:
        with pytest.raises(ValueError): h.validate_restore_args(args, root, iteration)


def scheduler_state():
    return dict(max_lr=0.01, min_lr=0., lr_warmup_steps=0, num_steps=64,
                lr_decay_style='constant', lr_decay_steps=128,
                start_wd=0., end_wd=0., wd_incr_style='constant', wd_incr_steps=128)


def rng_state():
    return dict(random_rng_state=random.getstate(), np_rng_state=np.random.get_state(),
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.arange(16, dtype=torch.uint8),
                rng_tracker_states={'model-parallel-rng': torch.arange(16, dtype=torch.uint8)})


@pytest.mark.parametrize('damage', ['none', 'scheduler', 'python', 'numpy', 'torch', 'cuda',
                                    'tracker', 'missing_scheduler', 'missing_rng', 'empty_tracker',
                                    'wrong_iteration', 'unknown_scheduler'])
def test_auxiliary_exact_check(damage):
    h = helper()
    actual = dict(scheduler=scheduler_state(), rng=rng_state())
    common = dict(iteration=3, opt_param_scheduler=copy.deepcopy(actual['scheduler']))
    payload = {'rng_state': [copy.deepcopy(actual['rng'])]}
    if damage == 'scheduler': actual['scheduler']['num_steps'] += 48
    if damage == 'python': actual['rng']['random_rng_state'] = random.Random(17).getstate()
    if damage == 'numpy': actual['rng']['np_rng_state'][1][100] ^= np.uint32(1)
    if damage in ('torch', 'cuda'): actual['rng'][damage + '_rng_state'][0] ^= 1
    if damage == 'tracker': actual['rng']['rng_tracker_states']['model-parallel-rng'][1] ^= 1
    if damage == 'missing_scheduler': del common['opt_param_scheduler']
    if damage == 'missing_rng': payload.clear()
    if damage == 'empty_tracker': actual['rng']['rng_tracker_states'].clear()
    if damage == 'wrong_iteration': common['iteration'] = 2
    if damage == 'unknown_scheduler': common['opt_param_scheduler'] = {'counter': 4}
    if damage == 'none':
        result = h.compare_auxiliary(actual, common, payload, 3)
        assert result['scheduler_exact'] and result['rng_exact']
        assert result['scheduler_num_steps'] == 64
        assert result['rng_source'] == 'DCP:rng_state'
    else:
        with pytest.raises(ValueError): h.compare_auxiliary(actual, common, payload, 3)


@pytest.mark.parametrize('delta', [1, 2, 3, 4])
def test_progress_allows_incident_window(delta):
    h = helper()
    model, opt = state_after_update()
    for _ in range(3): update(model, opt)
    before = h.fingerprint(state_dict(model, opt))
    for _ in range(delta): update(model, opt)
    after = h.fingerprint(state_dict(model, opt))
    h.validate_progress(before, after, step_delta=delta)
    assert after['optimizer_steps'] == [float(4 + delta)] * 2
    with pytest.raises(ValueError): h.validate_progress(before, after, step_delta=delta + 1)


def update(model, optimizer):
    optimizer.zero_grad()
    model(torch.ones(2, 3)).square().sum().backward()
    optimizer.step()


@pytest.fixture
def native_scene(tmp_path, monkeypatch):
    """Real CPU Adam + DCP; only Megatron/CUDA interfaces are substituted."""
    h = helper()
    h._RESTORE_CONTEXT.clear()
    root, args = scene_args(tmp_path)
    source, source_opt = state_after_update()
    for _ in range(3): update(source, source_opt)
    saved = copy.deepcopy(state_dict(source, source_opt))
    selected = root / 'actor/iter_0000003'
    dcp.save(saved, checkpoint_id=selected)
    saved_rng = rng_state()
    sched_state = scheduler_state()
    common = dict(iteration=3, opt_param_scheduler=copy.deepcopy(sched_state))
    torch.save(common, selected / 'common.pt')
    # Synthetic CPU RNG shard models the native ShardedObject's DCP read result.
    torch.save([saved_rng], selected / 'rng.pt')
    model, opt = state_after_update()
    scheduler = SimpleNamespace(state=scheduler_state())
    scheduler.state['num_steps'] = 0
    scheduler.state_dict = lambda: copy.deepcopy(scheduler.state)
    cuda_state = {'value': torch.full((16,), 99, dtype=torch.uint8)}
    tracker = {'states': {'model-parallel-rng': torch.full((16,), 99, dtype=torch.uint8)}}
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: cuda_state['value'].clone())
    tp = SimpleNamespace(get_cuda_rng_tracker=lambda: SimpleNamespace(get_states=lambda: tracker['states']))
    for name in ('megatron', 'megatron.training', 'megatron.core', 'benchmark'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    cp = ModuleType('megatron.training.checkpointing')
    dc = ModuleType('megatron.core.dist_checkpointing')
    cp.tensor_parallel = tp
    mpu = SimpleNamespace(**{name: (lambda *a, **kw: 1) for name in (
        'get_tensor_model_parallel_world_size', 'get_pipeline_model_parallel_world_size',
        'get_context_parallel_world_size', 'get_data_parallel_world_size',
        'get_expert_model_parallel_world_size')})
    monkeypatch.setattr(sys.modules['megatron.core'], 'mpu', mpu, raising=False)
    monkeypatch.setattr(sys.modules['megatron.core'], 'tensor_parallel', tp, raising=False)
    events, calls = [], []
    evidence = ModuleType('benchmark.runtime_evidence')
    evidence.emit = lambda a, event, **fields: events.append(dict(event=event, role=a.bench_role, **fields))
    for module in (cp, dc, evidence): monkeypatch.setitem(sys.modules, module.__name__, module)

    def generate(args, model, optimizer, opt_param_scheduler, rng_state, optim_sd_kwargs=None):
        calls.append(('generate', optim_sd_kwargs.copy()))
        return dict(state_dict(model, optimizer), rng_state=rng_state,
                    opt_param_scheduler=opt_param_scheduler.state_dict())

    def read(sharded_state_dict, checkpoint_dir, sharded_strategy=None, **kw):
        calls.append(('read', str(checkpoint_dir), sharded_strategy, kw.copy()))
        native_rng = sharded_state_dict.pop('rng_state', None)
        sharded_state_dict.pop('opt_param_scheduler', None)
        dcp.load(sharded_state_dict, checkpoint_id=checkpoint_dir)
        if native_rng is not None:
            sharded_state_dict['rng_state'] = torch.load(Path(checkpoint_dir) / 'rng.pt', weights_only=False)
        return sharded_state_dict

    cp.generate_state_dict, dc.load = generate, read
    monkeypatch.setenv('BENCH_RESTORE_ROOT', str(root))
    monkeypatch.setenv('BENCH_RESTORE_ITERATION', '3')
    flags = {}

    def load(model, optimizer, sched, **kw):
        template = cp.generate_state_dict(args, model, optimizer, sched, [rng_state()],
            optim_sd_kwargs={'is_loading': True, 'metadata': {'format': 'dp_reshardable'}})
        # Adam's independent native read target already has its moment topology.
        native_path = flags.get('native_path', Path(args.load) / 'iter_0000003')
        native = dc.load(h.cpu_template(template), native_path, sharded_strategy='cpu-test',
                         validate_access_integrity=False, strict='assume_ok_unexpected')
        model.load_state_dict(native['model'])
        optimizer.load_state_dict(native['optimizer'])
        if not flags.get('skip_scheduler'): sched.state = copy.deepcopy(common['opt_param_scheduler'])
        if not flags.get('skip_rng'):
            random.setstate(saved_rng['random_rng_state'])
            np.random.set_state(saved_rng['np_rng_state'])
            torch.set_rng_state(saved_rng['torch_rng_state'])
            cuda_state['value'] = saved_rng['cuda_rng_state'].clone()
            tracker['states'] = copy.deepcopy(saved_rng['rng_tracker_states'])
        if flags.get('damage_model'):
            with torch.no_grad(): model.weight[-1, -1] += 1
        if flags.get('wrong_counter'):
            for st in optimizer.state.values(): st['step'].add_(1)
        if flags.get('native_error'): raise RuntimeError('native load failed')
        return (flags.get('iteration', 3), 12345)

    initial_random, initial_np, initial_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    scene = SimpleNamespace(h=h, root=root, args=args, selected=selected, model=model, optimizer=opt,
        scheduler=scheduler, load=load, cp=cp, dc=dc, generate=generate, reader=read,
        events=events, calls=calls, flags=flags, saved=saved, saved_rng=saved_rng)
    yield scene
    random.setstate(initial_random)
    np.random.set_state(initial_np)
    torch.set_rng_state(initial_torch)
    h._RESTORE_CONTEXT.clear()


def restore(scene):
    return scene.h.restore_load_checkpoint(scene.args, scene.load, scene.model,
                                           scene.optimizer, scene.scheduler,
                                           checkpointing_context={}, skip_load_to_model_and_opt=False)


@pytest.mark.parametrize('role', ['actor', 'critic'])
def test_worker_native_readback_and_auxiliary_evidence(native_scene, role):
    s = native_scene
    if role == 'critic':
        import shutil
        shutil.copytree(s.selected, s.root / 'critic/iter_0000003', dirs_exist_ok=True)
        s.args.bench_role = 'critic'
        s.args.load, s.args.save = s.args.critic_load, s.args.critic_save
    assert restore(s) == (3, 12345)
    event = s.events[0]
    assert event['role'] == role and event['event'] == 'restore_loaded'
    assert event['loaded_iteration'] == 3 and event['next_rollout_id'] == 4
    assert event['snapshot']['optimizer_steps'] == [4., 4.]
    assert event['payload_exact'] and event['scheduler_exact'] and event['rng_exact']
    assert event['snapshot'] == s.h.fingerprint(s.saved)
    assert s.cp.generate_state_dict is s.generate and s.dc.load is s.reader
    assert [c[2:] for c in s.calls if c[0] == 'read'] == [
        ('cpu-test', {'validate_access_integrity': False, 'strict': 'assume_ok_unexpected'})] * 2
    assert [c[1]['is_loading'] for c in s.calls if c[0] == 'generate'] == [True, False, False]
    pickle.dumps(s.args)


@pytest.mark.parametrize('damage', ['skip_scheduler', 'skip_rng', 'damage_model', 'wrong_counter',
                                    'native_error', 'iteration', 'missing_common'])
def test_worker_rejects_false_restore_and_restores_hooks(native_scene, damage):
    s = native_scene
    s.flags[damage] = 2 if damage == 'iteration' else True
    if damage == 'missing_common': (s.selected / 'common.pt').unlink()
    with pytest.raises((ValueError, RuntimeError)):
        restore(s)
    assert not s.events
    assert s.cp.generate_state_dict is s.generate and s.dc.load is s.reader
    assert s.args.bench_role not in s.h._RESTORE_CONTEXT


def test_updates_5_to_8_and_final_save_delta_four(native_scene):
    s = native_scene
    restore(s)
    for rollout in range(4, 8):
        update(s.model, s.optimizer)
        s.scheduler.state['num_steps'] += 16
        s.h.restore_after_optimizer_step(s.args, s.optimizer, rollout, step_id=0)
    target = Path(s.args.save) / 'iter_0000007'
    dcp.save(copy.deepcopy(state_dict(s.model, s.optimizer)), checkpoint_id=target)
    s.h.restore_after_save(s.args, 7)
    progress = [r for r in s.events if r['event'] == 'restore_optimizer_step']
    assert [r['optimizer_steps'] for r in progress] == [[float(i)] * 2 for i in range(5, 9)]
    assert s.events[-1]['saved_iteration'] == 7
    assert s.events[-1]['step_delta'] == 4
    assert s.events[-1]['snapshot']['optimizer_steps'] == [8., 8.]


def test_save_hook_does_not_require_per_step_hook(native_scene):
    s = native_scene
    restore(s)
    for _ in range(4): update(s.model, s.optimizer)
    target = Path(s.args.save) / 'iter_0000007'
    dcp.save(copy.deepcopy(state_dict(s.model, s.optimizer)), checkpoint_id=target)
    s.h.restore_after_save(s.args, 7)
    assert s.events[-1]['step_delta'] == 4


def test_counter_hook_rejects_skipped_or_repeated_update(native_scene):
    s = native_scene
    restore(s)
    with pytest.raises(ValueError): s.h.restore_after_optimizer_step(s.args, s.optimizer, 4)
    update(s.model, s.optimizer)
    s.h.restore_after_optimizer_step(s.args, s.optimizer, 4)
    with pytest.raises(ValueError): s.h.restore_after_optimizer_step(s.args, s.optimizer, 4)


def test_passthrough_without_restore_environment(monkeypatch):
    h = helper()
    monkeypatch.delenv('BENCH_RESTORE_ROOT', raising=False)
    monkeypatch.delenv('BENCH_RESTORE_ITERATION', raising=False)
    marker = object()
    assert h.restore_load_checkpoint(None, lambda *a, **kw: marker, None, None, None) is marker
    h.restore_after_save(None, 0)


def test_partial_environment_cannot_select_fresh_fallback(monkeypatch):
    h = helper()
    monkeypatch.delenv('BENCH_RESTORE_ROOT', raising=False)
    monkeypatch.setenv('BENCH_RESTORE_ITERATION', '3')
    with pytest.raises(ValueError):
        h.restore_load_checkpoint(None, lambda *a, **kw: pytest.fail('must not load'), None, None, None)


def test_final_diagnostic_generation_must_preserve_rng(native_scene, monkeypatch):
    s = native_scene
    calls = 0
    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = s.generate(*args, **kwargs)
        if calls == 3:
            torch.rand(1)
        return result
    # Preserve the native signature used to force is_loading=False on snapshots.
    import functools
    generate = functools.wraps(s.generate)(generate)
    monkeypatch.setattr(s.cp, 'generate_state_dict', generate)
    with pytest.raises(ValueError, match='scheduler/RNG'):
        restore(s)
    assert not s.events


def test_worker_critic_identity_and_own_checkpoint(native_scene):
    import shutil
    s = native_scene
    target = s.root / 'critic/iter_0000003'
    shutil.copytree(s.selected, target, dirs_exist_ok=True)
    s.args.bench_role = 'critic'
    s.args.load, s.args.save = s.args.critic_load, s.args.critic_save
    s.flags['native_path'] = s.selected
    # The native seam's read path is bound to the actor folder, so it must fail,
    # even though both checkpoints contain byte-identical synthetic tensors.
    with pytest.raises(ValueError, match='different checkpoint'):
        restore(s)
    assert not s.events


@pytest.mark.parametrize('field', ['no_load_rng', 'finetune', 'no_load_optim', 'no_save_rng'])
def test_worker_rechecks_fallback_flags_before_native_read(native_scene, field):
    s = native_scene
    setattr(s.args, field, True)
    with pytest.raises(ValueError, match=field): restore(s)
    assert not s.calls and not s.events


def test_readback_mutation_is_detected(native_scene, monkeypatch):
    s = native_scene
    reads = 0
    def read(*args, **kwargs):
        nonlocal reads
        reads += 1
        value = s.reader(*args, **kwargs)
        if reads == 2:
            with torch.no_grad(): s.model.weight.add_(1)
        return value
    import functools
    read = functools.wraps(s.reader)(read)
    monkeypatch.setattr(s.dc, 'load', read)
    with pytest.raises(ValueError, match='modified live training state'): restore(s)
    assert not s.events


def test_pinned_native_scheduler_and_rng_mapping_on_cpu(native_scene, monkeypatch):
    """Optional cached source conformance; no Megatron installation/GPU import."""
    import ast
    import hashlib
    import logging
    import os
    import tarfile
    from typing import Optional
    archive = os.environ.get('INCIDENT_MEGATRON_SOURCE_ARCHIVE')
    if not archive:
        pytest.skip('Set INCIDENT_MEGATRON_SOURCE_ARCHIVE for cached pinned-source conformance')
    revision = '37f91617b95c1ef4f01fc66210e502572ad97fda'
    with tarfile.open(archive) as tar:
        prefix = 'Megatron-LM-' + revision + '/'
        scheduler_source = tar.extractfile(prefix + 'megatron/core/optimizer_param_scheduler.py').read().decode()
        mapping_source = tar.extractfile(prefix + 'megatron/core/dist_checkpointing/mapping.py').read().decode()
        utils_source = tar.extractfile(prefix + 'megatron/core/dist_checkpointing/dict_utils.py').read().decode()
    assert hashlib.sha256(mapping_source.encode()).hexdigest() == 'd0d62509655b2b788a1783bb39fedcb9d689340ffbfb971b87e5db7d4b247062'
    sched_class = next(n for n in ast.parse(scheduler_source).body if isinstance(n, ast.ClassDef) and n.name == 'OptimizerParamScheduler')
    namespace = dict(math=__import__('math'), logging=logging, Optional=Optional, logger=logging.getLogger(__name__),
                     log_single_rank=lambda *a, **kw: None)
    exec(compile(ast.Module(body=[sched_class], type_ignores=[]), '<pinned-scheduler>', 'exec'), namespace)
    s = native_scene
    scheduler = namespace['OptimizerParamScheduler'](optimizer=s.optimizer, init_lr=0., max_lr=0.01,
        min_lr=0., lr_warmup_steps=0, lr_decay_steps=128, lr_decay_style='constant',
        start_wd=0., end_wd=0., wd_incr_steps=128, wd_incr_style='constant',
        use_checkpoint_opt_param_scheduler=True, override_opt_param_scheduler=False)
    scheduler.load_state_dict(scheduler_state())
    assert scheduler.state_dict() == scheduler_state()
    actual = dict(scheduler=scheduler.state_dict(), rng=rng_state())
    common = dict(iteration=3, opt_param_scheduler=scheduler_state())
    assert s.h.compare_auxiliary(actual, common, {'rng_state': [actual['rng']]}, 3)['scheduler_exact']
    scheduler.step(increment=3 * s.args.global_batch_size)
    actual['scheduler'] = scheduler.state_dict()
    with pytest.raises(ValueError, match='scheduler differs'):
        s.h.compare_auxiliary(actual, common, {'rng_state': [actual['rng']]}, 3)

    # Execute the pinned pure mapping module with only its relative dependencies supplied.
    utils = {}
    exec(compile(utils_source, '<pinned-dict-utils>', 'exec'), utils)
    module = ModuleType('megatron.core.dist_checkpointing.mapping')
    module.CheckpointingException = RuntimeError
    module.dict_list_map_inplace = utils['dict_list_map_inplace']
    tree = ast.parse(mapping_source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level == 1 and n.module in ('core', 'dict_utils'))]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(tree, '<pinned-mapping>', 'exec'), module.__dict__)
    rng = module.ShardedObject('rng_state', [rng_state()], (1, 1), (0, 0), replica_id=0)
    assert rng.unique_key == 'rng_state/shard_0.0_1.1'
    copied = s.h.cpu_template({'rng_state': rng})['rng_state']
    copied.data[0]['np_rng_state'][1][10] ^= np.uint32(1)
    assert copied.data[0]['np_rng_state'][1][10] != rng.data[0]['np_rng_state'][1][10]
    copied.data[0]['torch_rng_state'][0] ^= 1
    assert copied.data[0]['torch_rng_state'][0] != rng.data[0]['torch_rng_state'][0]


def test_matching_disk_and_live_counter_five_is_not_step_three_restore(native_scene):
    s = native_scene
    wrong = copy.deepcopy(s.saved)
    for state in wrong['optimizer']['state'].values():
        state['step'].fill_(5)
    dcp.save(wrong, checkpoint_id=s.selected)
    # The native loader and independent read agree on all bytes. The scene's
    # iteration-to-Adam-counter contract must still reject this checkpoint.
    with pytest.raises(ValueError, match='must restore Adam counter 4'):
        restore(s)
    assert not s.events
