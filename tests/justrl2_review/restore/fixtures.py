# SPDX-License-Identifier: Apache-2.0
"""Real CPU Adam/DCP fixture; only Megatron/CUDA interfaces are substituted.

Extracted from the contribution's restore tests. No test functions or model
simulation: torch.nn.Linear, Adam, and filesystem DCP own all tensor state.
"""
import copy
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from tests.justrl2_review.fixtures import SOURCE, load


def helper():
    return load(SOURCE / 'restore_evidence.py')


def state_after_update():
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).square().sum().backward()
    opt.step()
    return model, opt


def state_dict(model, opt):
    return {'model': model.state_dict(), 'optimizer': opt.state_dict()}


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


def scheduler_state():
    return dict(max_lr=0.01, min_lr=0., lr_warmup_steps=0, num_steps=64,
                lr_decay_style='constant', lr_decay_steps=128,
                start_wd=0., end_wd=0., wd_incr_style='constant', wd_incr_steps=128)


def rng_state():
    return dict(random_rng_state=random.getstate(), np_rng_state=np.random.get_state(),
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.arange(16, dtype=torch.uint8),
                rng_tracker_states={'model-parallel-rng': torch.arange(16, dtype=torch.uint8)})


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
