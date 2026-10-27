"""Check real updates and independently deserialize complete checkpoints."""
import gc
import json
import math
from pathlib import Path


def rows(root, name):
    path = root / (name + '.jsonl')
    assert path.is_file(), 'missing runtime evidence: ' + name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_updates(root, first, last, budget):
    metrics = rows(root, 'metrics')
    updates = [r for r in metrics if 'actor/optimizer_step_min' in r]
    assert [r['step'] for r in updates] == list(range(first, last + 1)), 'incomplete trainer steps'
    assert all(r['actor/optimizer_step_min'] == r['actor/optimizer_step_max'] == r['step'] for r in updates), 'optimizer step mismatch'
    assert all(math.isfinite(r['actor/grad_norm']) for r in updates), 'nonfinite gradients'
    events = rows(root, 'worker_events')
    assert [e['optimizer_step'] for e in events] == list(range(first, last + 1))
    assert all(sum(e['optimizer_state_bytes'].values()) >= 3_900_000_000 for e in events), 'incomplete Adam state'
    assert 'val/test_score/openai/gsm8k' in metrics[0], 'missing initial validation'
    assert json.loads((root/'completed.json').read_text())['steps'] == last
    ckpt = rows(root, 'checkpoint_events')
    assert all(e['host_memory_limit_bytes'] == budget for e in ckpt), 'host memory budget changed'
    counters = dict(line.split() for line in (root/'final_memory_events.txt').read_text().splitlines())
    assert int(counters.get('oom_kill', 0)) == 0, 'kernel OOM kill during training'
    return updates, ckpt


def check_checkpoint_failure(root, stderr, budget):
    """Require actual updates followed by host-memory eviction inside save."""
    assert [r['step'] for r in rows(root, 'metrics')] == list(range(10))
    assert [r['optimizer_step'] for r in rows(root, 'worker_events')] == list(range(1, 11))
    events = rows(root, 'checkpoint_events')
    assert events and events[0]['phase'] == 'save_start' and events[0]['step'] == 10
    assert all(e['step'] == 10 and e['phase'] != 'save_complete' for e in events)
    assert all(e['host_memory_limit_bytes'] == budget for e in events)
    assert 'Task was killed due to the node running low on memory' in stderr
    assert '_save_checkpoint' in stderr and 'save_checkpoint' in stderr
    assert 'CUDA out of memory' not in stderr
    assert not (root/'completed.json').exists()


def check_checkpoint(root, step, model_path):
    import torch
    from safetensors import safe_open
    directory = root/'checkpoints'/f'global_step_{step}'
    actor = directory/'actor'
    model = torch.load(actor/'model_world_size_1_rank_0.pt', map_location='cpu', weights_only=True)
    with safe_open(str(model_path/'model.safetensors'), framework='pt', device='cpu') as initial:
        expected = set(initial.keys())
        # Tied output weights may be omitted in safetensors but appear in FSDP state.
        assert expected <= set(model), 'missing checkpoint parameters'
        for name in expected:
            assert tuple(model[name].shape) == tuple(initial.get_slice(name).get_shape()), name
    assert sum(t.numel() for t in model.values()) >= 490_000_000
    assert all(torch.isfinite(t).all().item() for t in model.values()), 'nonfinite saved model'
    probes = {k: v.flatten()[::max(1, v.numel()//64)][:64].clone() for k,v in model.items()}
    del model
    gc.collect()
    optimizer = torch.load(actor/'optim_world_size_1_rank_0.pt', map_location='cpu', weights_only=True)
    assert optimizer['param_groups'] and optimizer['state']
    total = 0
    for state in optimizer['state'].values():
        assert int(state['step']) == step, 'wrong checkpoint optimizer step'
        assert state['exp_avg'].shape == state['exp_avg_sq'].shape
        for key in ['exp_avg','exp_avg_sq']:
            value = state[key]
            assert value.dtype == torch.float32, 'optimizer checkpoint precision changed'
            assert torch.isfinite(value).all().item(), 'nonfinite saved optimizer'
            total += value.numel() * value.element_size()
    assert total >= 3_900_000_000, 'incomplete serialized Adam moments'
    del optimizer
    gc.collect()
    extra = torch.load(actor/'extra_state_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
    assert extra['lr_scheduler']['last_epoch'] == step
    assert extra['rng'], 'missing RNG state'
    assert (directory/'data.pt').is_file(), 'missing dataloader state'
    return {'step':step,'optimizer_moment_bytes':total}, probes
