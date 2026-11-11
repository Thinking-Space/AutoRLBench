#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute a configured sequence of native CPU replay updates.

This runs function-level replay with synthetic critic values, not model training.
The run records completed-update summaries and its first reported exception.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import torch
from types import SimpleNamespace

import run_replay


def default_config():
    return dict(cp=2, dp=2, order=[0, 1, 2, 3], updates=5, copies=2,
                seed=3, assignment='shuffle', reporting=True,
                compute_timeout=4.0, report_timeout=1.0, outer_timeout=45.0)


def validate_config(config):
    if set(config) != set(default_config()):
        raise ValueError('Unexpected or missing replay configuration keys')
    run_replay._configuration(config['cp'], config['dp'], config['order'], [3], 'ones', config['seed'])
    if config['dp'] != 2 or config['copies'] != 2 or type(config['copies']) is not int:
        raise ValueError('This input schedule requires two replicas and two copies')
    if type(config['updates']) is not int or not 1 <= config['updates'] <= 12:
        raise ValueError('updates must be 1..12')
    if config['assignment'] not in ('shuffle', 'interleave') or type(config['reporting']) is not bool:
        raise ValueError('Invalid assignment or reporting setting')
    for key in ('compute_timeout', 'report_timeout', 'outer_timeout'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or not 0 < config[key] <= 60:
            raise ValueError('Timeouts must be positive finite seconds <=60')
    return config


def assignment(config, update):
    ids = list(range(4))
    if config['assignment'] == 'shuffle':
        random.Random(config['seed'] + update).shuffle(ids)
    else:
        ids = [0, 2, 1, 3]
    return [ids[:2], ids[2:]]


def inputs(source):
    rows = run_replay._fixture_samples(Path(source) / 'inputs/replay.json', 2)
    return [rows[0], rows[0], rows[1], rows[1]]


def input_summary(pool):
    lengths = [row['length'] for row in pool]
    return dict(samples=len(pool), response_tokens=sum(lengths),
                response_min=min(lengths), response_max=max(lengths),
                reward_mean=sum(row['reward'] for row in pool) / len(pool))


def metric_contribution(sample_rows, *, cp_size):
    """Per-rank native log_dict count and CP-local sum of per-sample means.

    Each row is (local metric tensor, local mask, full per-sample mask).
    The denominator is the full sample mask count and is replicated on each
    CP rank; only the numerator is sliced to this rank's CP coordinates.
    """
    if type(cp_size) is not int or cp_size < 1 or not sample_rows:
        raise ValueError('invalid metric contribution')
    numerator = 0.0
    for values, local_mask, full_mask in sample_rows:
        if values.numel() != local_mask.numel():
            raise ValueError('local values/mask lengths differ')
        numerator += float((values * local_mask).sum() / full_mask.sum().clamp_min(1))
    return torch.tensor([len(sample_rows), numerator], dtype=torch.float64)


def native_report_reference(rank_contributions, *, cp_size):
    """Mirror aggregate_train_losses' SUM and CP correction for one metric."""
    if not rank_contributions:
        raise ValueError('rank contributions cannot be empty')
    total = torch.stack(rank_contributions).sum(dim=0)
    if total[0] <= 0:
        raise ValueError('global sample denominator must be positive')
    return (total[1:] * cp_size / total[0]).tolist()


def configure_loader():
    run_replay.NATIVE_FUNCTIONS['miles/backends/training_utils/log_utils.py'] = ('aggregate_train_losses',)
    key = 'miles/backends/training_utils/cp_utils.py'
    if 'get_local_response_loss_masks' not in run_replay.NATIVE_FUNCTIONS[key]:
        run_replay.NATIVE_FUNCTIONS[key] += ('get_local_response_loss_masks',)


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker(source, output, scratch, config, rank):
    code, dist = 2, None
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        torch.set_num_threads(1)
        configure_loader()
        world = config['cp'] * config['dp']
        order = config['order']
        dp_rank, cp_rank = divmod(order.index(rank), config['cp'])
        dist.init_process_group('gloo', init_method=(scratch / 'rendezvous').as_uri(),
                                world_size=world, rank=rank, timeout=timedelta(seconds=15))
        control = dist.new_group(list(range(world)), timeout=timedelta(seconds=60))
        cp_group = dp_group = None
        for d in range(config['dp']):
            members = order[d * config['cp']:(d + 1) * config['cp']]
            group = dist.new_group(members, timeout=timedelta(seconds=config['compute_timeout']))
            if rank in members:
                cp_group = group
        for c in range(config['cp']):
            members = order[c::config['cp']]
            group = dist.new_group(members, timeout=timedelta(seconds=config['compute_timeout']))
            if rank in members:
                dp_group = group
        reporting_group = dist.new_group(list(range(world)), timeout=timedelta(seconds=config['report_timeout']))
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=config['cp'], group=cp_group),
                                intra_dp=SimpleNamespace(rank=dp_rank, size=config['dp'], group=dp_group),
                                intra_dp_cp=SimpleNamespace(group=reporting_group, size=world, rank=rank))
        logger = logging.getLogger('sequence_replay')
        tracking = SimpleNamespace(log=lambda *args, **kwargs: None)
        namespace = dict(torch=torch, dist=dist, F=F, get_parallel_state=lambda: state,
                         Namespace=SimpleNamespace, RolloutBatch=dict, logger=logger,
                         tracking_utils=tracking, compute_rollout_step=lambda args, rollout_id: rollout_id)
        ready = 0
        try:
            run_replay._load_native(source, namespace)
            pool = inputs(source)
        except Exception:
            ready = 1
        flag = torch.tensor([ready]); dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control)
        if flag.item():
            return 2
        settings = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, advantage_estimator='ppo',
                                   gamma=1.0, lambd=1.0, gae_lambda_k=0.5, normalize_advantages=True, qkv_format='thd')
        completed = []
        code = 0
        for update in range(config['updates']):
            selected = assignment(config, update)[dp_rank]
            samples = [pool[i] for i in selected]
            record = dict(update=update, rank=rank, sample_ids=selected,
                          samples=[dict(total=x['total'], length=x['length'], reward=x['reward'],
                                        masks=x['masks'], values=x['values']) for x in samples],
                          status='started', whitening_called=False, whitening_members=None,
                          whitening_values=None, whitening_mask=None, advantages=None, returns=None)
            failed = 0
            operation = 'compute_advantages_and_returns'
            try:
                values = [namespace['slice_log_prob_with_cp'](torch.tensor(s['values']), s['total'], s['length']) for s in samples]
                batch = dict(values=values, log_probs=[torch.zeros_like(x) for x in values],
                             rewards=[s['reward'] for s in samples], total_lengths=[s['total'] for s in samples],
                             response_lengths=[s['length'] for s in samples], loss_masks=[torch.tensor(s['masks'], dtype=torch.float32) for s in samples])
                record['local_values'] = [x.tolist() for x in values]
                loss_module = sys.modules['miles.backends.training_utils.loss']
                native_whiten = loss_module.distributed_masked_whiten
                def capture_whiten(advs, masks, process_group=None, **kwargs):
                    record['whitening_called'] = True
                    record['whitening_members'] = dist.get_process_group_ranks(process_group)
                    record['whitening_values'] = advs.detach().float().cpu().tolist()
                    record['whitening_mask'] = masks.detach().float().cpu().tolist()
                    try:
                        result = native_whiten(advs, masks, process_group=process_group, **kwargs)
                        record['whitening_output'] = result.detach().float().cpu().tolist()
                        return result
                    except Exception as exc:
                        record['whitening_error_type'] = type(exc).__name__
                        raise
                loss_module.distributed_masked_whiten = capture_whiten
                native_gae = loss_module.get_advantages_and_returns_batch
                def capture_gae(*args, **kwargs):
                    advantages, returns = native_gae(*args, **kwargs)
                    record['raw_advantages'] = [x.detach().float().cpu().tolist() for x in advantages]
                    record['raw_returns'] = [x.detach().float().cpu().tolist() for x in returns]
                    return advantages, returns
                loss_module.get_advantages_and_returns_batch = capture_gae
                namespace['compute_advantages_and_returns'](settings, batch)
                loss_module.distributed_masked_whiten = native_whiten
                loss_module.get_advantages_and_returns_batch = native_gae
                record.update(status='computed',
                              advantages=[x.detach().float().cpu().tolist() for x in batch['advantages']],
                              returns=[x.detach().float().cpu().tolist() for x in batch['returns']])
                write_json(scratch / f'evidence-{rank}-{update}.json', record)
                operation = 'aggregate_train_losses'
                metrics = None
                if config['reporting']:
                    local_masks = namespace['get_local_response_loss_masks'](
                        batch['total_lengths'], batch['response_lengths'],
                        batch['loss_masks'], settings.qkv_format,
                    )
                    adv_rows = list(zip([v.square() for v in batch['advantages']], local_masks, batch['loss_masks'], strict=True))
                    ret_rows = list(zip([v.square() for v in batch['returns']], local_masks, batch['loss_masks'], strict=True))
                    adv_part = metric_contribution(adv_rows, cp_size=config['cp'])
                    ret_part = metric_contribution(ret_rows, cp_size=config['cp'])
                    numerator = torch.tensor(
                        [len(samples), adv_part[1].item(), ret_part[1].item()], dtype=torch.float32
                    )
                    metrics = namespace['aggregate_train_losses']([
                        dict(keys=['adv_sample_mean_square', 'return_sample_mean_square'], values=numerator)
                    ])
                    record['report_metrics'] = metrics
                    write_json(scratch / f'evidence-{rank}-{update}.json', record)
                if rank == 0:
                    completed.append(dict(update=update, input_summary=input_summary(pool), metrics=metrics))
            except Exception as exc:
                failed = 1
                # Preserve every peer error for reviewer audit.
                error = dict(rank=rank, update=update, operation=operation,
                             type=type(exc).__name__, message=str(exc), observed_at_ns=time.monotonic_ns())
                record.update(status='compute_error' if operation == 'compute_advantages_and_returns' else 'aggregate_error', error_type=type(exc).__name__)
                write_json(scratch / f'evidence-{rank}-{update}.json', record)
                write_json(scratch / f'error{rank}.json', error)
            flag = torch.tensor([failed]); dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control)
            if flag.item():
                if rank == 0 and completed and completed[-1]['update'] == update:
                    completed.pop()
                code = 1
                break
            if rank == 0:
                write_json(output / 'history.json', completed)
    except Exception:
        code = 2
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                code = 2
    return code


def run_sequence(source, output, config=None):
    config = validate_config(dict(default_config() if config is None else config))
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_loader()
    result = dict(status='infrastructure_error', exit_code=2, completed_updates=0, first_error=None,
                  source_sha256={}, outer_timeout=False)
    write_json(output / 'config.json', config)
    write_json(output / 'history.json', [])
    processes, streams = [], []
    try:
        inputs(source)
        _, before = run_replay._source_functions(source)
        result['source_sha256'] = before
        with tempfile.TemporaryDirectory(prefix='.sequence-', dir=output) as temp:
            scratch = Path(temp)
            deadline = time.monotonic() + config['outer_timeout']
            try:
                for rank in range(config['cp'] * config['dp']):
                    stream = (output / f'rank{rank}.stdout.log').open('w'); streams.append(stream)
                    processes.append(subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                        '--source', str(source), '--output', str(output), '--config', str(output / 'config.json'),
                        '--_worker', str(rank), '--_scratch', str(scratch)], stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_DISTRIBUTED_DEBUG='OFF', OMP_NUM_THREADS='1')) )
                while any(p.poll() is None for p in processes):
                    if time.monotonic() > deadline:
                        result['outer_timeout'] = True
                        break
                    time.sleep(0.02)
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                for p in processes:
                    try:
                        p.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        p.kill(); p.wait()
                for stream in streams:
                    stream.close()
            errors = [json.loads(p.read_text()) for p in scratch.glob('error*.json')]
            if errors:
                result['first_error'] = dict(min(errors, key=lambda x: x['observed_at_ns']))
                result['first_error'].pop('observed_at_ns')
            _, after = run_replay._source_functions(source)
            codes = [p.returncode for p in processes]
            if before == after and not result['outer_timeout'] and len(codes) == config['cp'] * config['dp'] and all(c in (0, 1) for c in codes):
                result.update(status='failed' if any(codes) else 'completed', exit_code=int(any(codes)))
            result['completed_updates'] = len(json.loads((output / 'history.json').read_text()))
            evidence = {}
            for path in scratch.glob('evidence-*.json'):
                row = json.loads(path.read_text())
                evidence.setdefault(row['update'], {})[str(row['rank'])] = row
            result['evidence'] = evidence
            result['errors'] = errors
    except Exception:
        pass
    write_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--_worker', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_scratch', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    if args._worker is not None:
        raise SystemExit(worker(args.source, args.output, args._scratch, configuration, args._worker))
    result = run_sequence(args.source, args.output, configuration)
    print(json.dumps(result))
    raise SystemExit(result['exit_code'])
