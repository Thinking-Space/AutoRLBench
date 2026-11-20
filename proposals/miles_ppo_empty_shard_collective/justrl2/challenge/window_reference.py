# SPDX-License-Identifier: Apache-2.0
"""Independent scalar prediction of aligned whitening and stale-identity runs."""
from __future__ import annotations
import math
import random
from sequence_audit import scalar_sample


def model_window(fixture, config, *, poisoned_identity=False):
    cp, dp, order = config['cp'], config['dp'], config['order']
    originals = fixture['samples']
    rows = []
    for update in range(config['updates']):
        ids = list(range(4))
        if config['assignment'] == 'shuffle':
            random.Random(config['seed'] + update).shuffle(ids)
        else:
            ids = [0, 2, 1, 3]
        samples = []
        for sid, item in enumerate([originals[0], originals[0], originals[1], originals[1]]):
            length = item['response_length']
            values = [.52 + .17 * update + .21 * math.sin(j / 11) + .03 * sid for j in range(length)]
            sample = dict(total=len(item['tokens']), length=length, masks=item['loss_mask'],
                          reward=item['reward'], values=values)
            sample.update(scalar_sample(sample, cp))
            samples.append(sample)
        peers = {}
        for d in range(dp):
            selected = ids[d * 2:(d + 1) * 2]
            for c in range(cp):
                rank = order[d * cp + c]
                local = lambda key: [[samples[sid][key][j] for j in samples[sid]['indices'][c]] for sid in selected]
                row = dict(update=update, rank=rank, sample_ids=selected,
                    local_values=local('values'), raw_advantages=local('advantages'),
                    raw_returns=local('returns'), local_masks=local('masks'),
                    full_masks=[samples[sid]['masks'] for sid in selected],
                    ppo_kl=[[.35 * math.sin(j / 17 + update * .43 + sid * .07)
                             for j in samples[sid]['indices'][c]] for sid in selected],
                    whitening_members=[order[other * cp + c] for other in range(dp)])
                row['whitening_values'] = [v for part in row['raw_advantages'] for v in part]
                row['whitening_mask'] = [v for part in row['local_masks'] for v in part]
                row['whitening_called'] = True
                row['advantages'] = row['raw_advantages']
                row['returns'] = row['raw_returns']
                peers[rank] = row
        rows.append(peers)

    identities = {rank: [0.0, 0.0, 0.0] for rank in range(cp * dp)}
    pairings = []
    for update, peers in enumerate(rows):
        for c in range(cp):
            ranks = [order[d * cp + c] for d in range(dp)]
            inputs = {}
            for rank in ranks:
                row = peers[rank]
                values, masks = row['whitening_values'], row['whitening_mask']
                identity = identities[rank]
                if values:
                    local = [sum(v * m for v, m in zip(values, masks)),
                             sum(v * v * m for v, m in zip(values, masks)), sum(masks)]
                    inputs[rank] = [a + b for a, b in zip(identity, local)]
                else:
                    inputs[rank] = list(identity)
                row['identity_before'] = list(identity)
            reduced = [sum(inputs[rank][axis] for rank in ranks) for axis in range(3)]
            count = reduced[2]
            if count == 0:
                raise ValueError('global mask sum is zero')
            mean = reduced[0] / count
            variance = reduced[1] / count - mean * mean
            if count >= 2:
                variance *= count / (count - 1)
            for rank in ranks:
                row = peers[rank]
                row['ordinal'] = update
                row['collective_input'] = inputs[rank]
                row['collective_output'] = list(reduced)
                if poisoned_identity and not row['whitening_values']:
                    identities[rank] = list(reduced)
                row['identity_after'] = list(identities[rank])
                row['advantages'] = [[(v - mean) / math.sqrt(variance + 1e-8) for v in chunk]
                                     for chunk in row['raw_advantages']]
                row['whitening_output'] = [v for chunk in row['advantages'] for v in chunk]
            pairings.append(dict(members=ranks, ordinal=update,
                                 updates=[update, update], mask_count=count))

    metrics = []
    for peers in rows:
        totals = dict(policy_loss=0., value_loss=0., clip_fraction=0.)
        for row in peers.values():
            policy, clipped = [], []
            for advs, rets, values, kl, mask, full in zip(
                    row['advantages'], row['returns'], row['local_values'], row['ppo_kl'],
                    row['local_masks'], row['full_masks'], strict=True):
                pg_part, clip_part = [], []
                denominator = max(sum(full), 1) * 2 * dp
                for adv, ret, value, logratio, weight in zip(advs, rets, values, kl, mask, strict=True):
                    ratio = math.exp(-logratio)
                    direct, bound = -ratio * adv, -min(1.2, max(.8, ratio)) * adv
                    pg, clip = max(direct, bound), float(bound > direct)
                    totals['policy_loss'] += pg * weight / denominator
                    totals['value_loss'] += (ret - value) ** 2 * weight / denominator
                    totals['clip_fraction'] += clip * weight / denominator
                    pg_part.append(pg); clip_part.append(clip)
                policy.append(pg_part); clipped.append(clip_part)
            row['policy_losses'], row['clip_flags'] = policy, clipped
        metrics.append(totals)
    return dict(rows=rows, metrics=metrics, pairings=pairings)


def exported_expectations(fixture):
    return {str(cp): model_window(fixture, dict(cp=cp, dp=2, order=list(range(cp * 2)),
                 updates=5, seed=3, assignment='shuffle'))['metrics'] for cp in (1, 2, 4)}


def validate_trace(fixture, config, traces, metrics, *, poisoned_identity=False):
    expected = model_window(fixture, config, poisoned_identity=poisoned_identity)
    errors = []
    world = config['cp'] * config['dp']
    def close(actual, target):
        if isinstance(target, list):
            return isinstance(actual, list) and len(actual) == len(target) and all(close(a, b) for a, b in zip(actual, target))
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, target, rel_tol=3e-4, abs_tol=3e-4)
    if set(traces) != set(map(str, range(world))):
        errors.append('missing or unexpected worker trace')
    for rank in range(world):
        actual = traces.get(str(rank), {}).get('rows', [])
        if len(actual) != config['updates']:
            errors.append(f'rank {rank}: missing updates')
            continue
        for update, observed in enumerate(actual):
            row = expected['rows'][update][rank]
            for key in ('local_values', 'raw_advantages', 'raw_returns', 'advantages', 'returns', 'policy_losses', 'clip_flags'):
                if not close(observed.get(key), row[key]):
                    errors.append(f'rank {rank} update {update}: {key} differs')
            if observed.get('update') != update or observed.get('whitening_called') is not True:
                errors.append(f'rank {rank} update {update}: participation differs')
            for key in ('ordinal', 'whitening_members'):
                if observed.get(key) != row[key]:
                    errors.append(f'rank {rank} update {update}: {key} differs')
            for key in ('whitening_values', 'whitening_mask', 'whitening_output',
                        'collective_input', 'collective_output'):
                if not close(observed.get(key), row[key]):
                    errors.append(f'rank {rank} update {update}: {key} differs')
        reports = traces.get(str(rank), {}).get('reports', [])
        if len(reports) != config['updates'] or any(
                not isinstance(report, dict) or set(report) != set(target)
                or any(not close(report.get(key), value) for key, value in target.items())
                for report, target in zip(reports, expected['metrics'])):
            errors.append(f'rank {rank}: report results differ or are missing')
        if traces.get(str(rank), {}).get('code') != 0:
            errors.append(f'rank {rank}: worker did not complete')
    if len(metrics) != config['updates']:
        errors.append('missing report metrics')
    else:
        for update, (observed, target) in enumerate(zip(metrics, expected['metrics'])):
            for key, value in target.items():
                if not close(observed.get(key), value):
                    errors.append(f'update {update}: report {key} differs')
    return dict(passed=not errors, errors=errors, pairings=expected['pairings'],
                complete_dataflow_checked=True, workers=world, updates=config['updates'])
