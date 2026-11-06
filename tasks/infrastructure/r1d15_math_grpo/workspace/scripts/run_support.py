"""Durable run state and observational metrics."""
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def state_hash(value):
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            if hasattr(x, 'to_local'):
                x = x.to_local()
            x = x.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def manager_audit(manager):
    return {
        'parameters': state_hash(dict(manager.model.named_parameters())),
        'optimizer': state_hash(manager.optimizer.state_dict()),
        'scheduler': state_hash(manager.lr_scheduler.state_dict()),
        'scaler': state_hash(manager.scaler.state_dict() if manager.scaler else None),
    }


def record_optimizer(actor, grad_norm):
    import torch
    rank = torch.distributed.get_rank()
    steps = [float(v['step'].item()) for v in actor.actor_optimizer.state.values() if 'step' in v]
    row = {'time': time.time(), 'rank': rank, 'attempt': os.environ['ATTEMPT'],
        'grad_norm': float(grad_norm.item()), 'finite_gradient': bool(torch.isfinite(grad_norm)),
        'optimizer_step_min': min(steps) if steps else 0, 'optimizer_step_max': max(steps) if steps else 0,
        'loss_scale': actor.scaler.get_scale() if actor.scaler else None}
    if not math.isfinite(row['grad_norm']): row['grad_norm'] = str(row['grad_norm'])
    with (Path(os.environ['RUN_DIR']) / f'optimizer_rank_{rank}.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def save_checkpoint(trainer):
    import numpy as np
    import torch
    root = Path(trainer.config.trainer.default_local_dir)
    root.mkdir(parents=True, exist_ok=True)
    step = trainer.global_steps
    stage = root / ('.pending_' + str(step) + '_' + str(os.getpid()))
    stage.mkdir(exist_ok=False)
    trainer.actor_rollout_wg.save_checkpoint(str(stage / 'actor'), None, step, max_ckpt_to_keep=None)
    torch.save(trainer.train_dataloader.state_dict(), stage / 'data.pt')
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'step': step}, stage / 'controller.pt')
    files = []
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
            files.append({'path': str(p.relative_to(stage)), 'bytes': p.stat().st_size, 'sha256': file_hash(p)})
    assert len(list((stage / 'actor').glob('model_world_size_8_rank_*.pt'))) == 8
    assert len(list((stage / 'actor').glob('optim_world_size_8_rank_*.pt'))) == 8
    assert len(list((stage / 'actor').glob('extra_state_world_size_8_rank_*.pt'))) == 8
    atomic_json(stage / 'complete.json', {'step': step, 'files': files, 'time': time.time()})
    final = root / ('global_step_' + str(step))
    assert not final.exists(), final
    os.replace(stage, final)
    atomic_json(root / 'latest.json', {'step': step, 'path': str(final)})
    (root / 'latest_checkpointed_iteration.txt').write_text(str(step))
    # Publish first, then rotate only this run's completed checkpoints.
    complete = sorted((p for p in root.glob('global_step_*') if (p / 'complete.json').exists()),
                      key=lambda p: int(p.name.split('_')[-1]))
    best = root / 'best_validation.json'
    protected = json.loads(best.read_text()).get('step') if best.exists() else None
    for old in complete[:-2]:
        if int(old.name.split('_')[-1]) != protected:
            shutil.rmtree(old)
    print('CHECKPOINT_COMMITTED', str(final), flush=True)


def validate_checkpoint(path):
    path = Path(path)
    d = json.loads((path / 'complete.json').read_text())
    for x in d['files']:
        p = path / x['path']
        if not p.is_file() or p.stat().st_size != x['bytes'] or file_hash(p) != x['sha256']:
            raise RuntimeError('Incomplete/corrupt checkpoint: ' + str(p))
    return d


def restore_controller(trainer, path):
    import numpy as np
    import torch
    d = torch.load(Path(path) / 'controller.pt', weights_only=False)
    assert d['step'] == trainer.global_steps
    random.setstate(d['python']); np.random.set_state(d['numpy']); torch.set_rng_state(d['torch'])
    atomic_json(Path(os.environ['RUN_DIR']) / 'resume.json', {'step': d['step'], 'time': time.time(),
        'data_state_loaded': True, 'controller_rng_loaded': True})


def parity_metrics(batch, step):
    import torch
    mask = batch.batch['response_mask'].bool()
    a = batch.batch['old_log_probs'].double()
    b = batch.batch['rollout_log_probs'].double()
    delta = a - b
    selected = delta[mask]
    assert torch.isfinite(selected).all(), 'Non-finite train/inference logprob'
    prob = (a.exp() - b.exp())[mask]
    length = mask.sum(-1).double()
    sums = torch.where(mask, delta, 0).sum(-1)
    weights = torch.exp(sums - sums.max())
    q = lambda x, p: torch.quantile(x, p).item()
    out = {'parity/tokens': mask.sum().item(), 'parity/sequences': mask.shape[0],
        'parity/logprob_abs_mean_nat': selected.abs().mean().item(),
        'parity/logprob_abs_max_nat': selected.abs().max().item(),
        'parity/logprob_signed_mean_nat': selected.mean().item(),
        'parity/prob_abs_mean': prob.abs().mean().item(), 'parity/prob_abs_max': prob.abs().max().item(),
        'parity/prob_signed_min': prob.min().item(), 'parity/prob_signed_max': prob.max().item(),
        'parity/k1_sample_estimate': -selected.mean().item(),
        'parity/seq_ess_fraction': (weights.sum().square() / weights.square().sum() / len(weights)).item(),
        'parity/mis_rejected_sequence_fraction': (sums > math.log(3)).double().mean().item(),
        'parity/sequence_log_ratio_min': sums.min().item(), 'parity/sequence_log_ratio_max': sums.max().item(),
        'sampling/length_mean': length.mean().item(), 'sampling/length_p95': q(length, .95),
        'sampling/cap_fraction': (length >= 8192).double().mean().item()}
    for p in [.5, .95, .99]:
        tag = str(round(p * 100))
        out['parity/logprob_abs_p' + tag + '_nat'] = q(selected.abs(), p)
        out['parity/sequence_log_ratio_p' + tag] = q(sums, p)
    for lo, hi in [(0, 1024), (1024, 4096), (4096, 8193)]:
        m = mask & ((length >= lo) & (length < hi))[:, None]
        if m.any(): out[f'parity/length_{lo}_{hi}_abs_mean_nat'] = delta[m].abs().mean().item()
    run = Path(os.environ['RUN_DIR']); record = run / 'parity_record.json'
    old = json.loads(record.read_text()) if record.exists() else {'max_nat': -1}
    if out['parity/logprob_abs_max_nat'] > old['max_nat']:
        # Observation evidence; checkpoint step N-1 is the matching pre-update state when available.
        idx = torch.where(mask, delta.abs(), -1).argmax().item()
        i, t = divmod(idx, mask.shape[1])
        atomic_json(record, {'step': step, 'max_nat': out['parity/logprob_abs_max_nat'],
            'sample': i, 'position': t, 'token': int(batch.batch['responses'][i,t]),
            'logp_actor': a[i,t].item(), 'logp_rollout': b[i,t].item(),
            'note': 'Native observation; no claim of bitwise decode replay'})
    return out


def append_metrics(data, step):
    run = Path(os.environ['RUN_DIR'])
    def clean(x):
        if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
        if hasattr(x, 'item'): x = x.item()
        if isinstance(x, float) and not math.isfinite(x): return str(x)
        return x
    row = {'step': int(step), 'time': time.time(), 'attempt': os.environ['ATTEMPT'], 'metrics': clean(data)}
    with (run / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
    atomic_json(run / 'latest_metrics.json', row)
    # AIME avg@32 is an external validation score, not the 1460-question training score.
    vals = {k:v for k,v in data.items() if k.startswith('val-core/') and 'mean@32' in k}
    if vals and step > 0:
        root = run / 'checkpoints'; p = root / 'best_validation.json'
        score = sum(float(v) for v in vals.values()) / len(vals)
        old = json.loads(p.read_text()) if p.exists() else {'score': -1e99}
        if score > old['score']:
            atomic_json(p, {'step': int(step), 'score': score, 'metrics': clean(vals)})
    if int(step) == 1 and os.environ.get('RESUME_PROBE') == '1':
        assert (run / 'checkpoints/global_step_1/complete.json').exists()
        atomic_json(run / 'resume_probe.saved.json', {'step': 1})
        # Ray propagates this exception; the supervisor accepts it only with the complete marker.
        raise RuntimeError('RESUME_PROBE_SAVED_RESTART_REQUIRED')
