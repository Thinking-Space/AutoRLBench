# SPDX-License-Identifier: Apache-2.0
"""Independent scalar audit for the maintainer's staged batch-assignment replay."""
from __future__ import annotations
import math
import random


def assignment(config, update):
    ids = list(range(config['copies'] * config['dp']))
    if config['assignment'] == 'shuffle':
        random.Random(config['seed'] + update).shuffle(ids)
    else:
        ids = [0, 2, 1, 3]
    return [ids[d * config['copies']:(d + 1) * config['copies']] for d in range(config['dp'])]


def response_indices(sample, cp_size, cp_rank):
    total, length = sample['total'], sample['length']
    prompt = total - length
    width = math.ceil(total / (2 * cp_size))
    result = []
    for token in range(length):
        block = (prompt - 1 + token) // width
        owner = min(block, 2 * cp_size - 1 - block) if cp_size > 1 else 0
        if owner == cp_rank:
            result.append(token)
    return result


def scalar_sample(sample, cp):
    length, reward, values = sample['length'], sample['reward'], sample['values']
    decay, carry = (0.5 ** (1 / length) if length else 0.0), 0.0
    advantages = [0.0] * length
    for index in reversed(range(length)):
        delta = (reward if index == length - 1 else 0.0)
        delta += (values[index + 1] if index + 1 < length else 0.0) - values[index]
        carry = delta + decay * carry
        advantages[index] = carry
    returns = [sum((reward if i == length - 1 else 0.0) for i in range(index, length))
               for index in range(length)]
    return dict(advantages=advantages, returns=returns,
                indices=[response_indices(sample, cp, rank) for rank in range(cp)])


def _close(actual, expected, *, atol=4e-4, rtol=4e-4):
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_close(a, e, atol=atol, rtol=rtol) for a, e in zip(actual, expected)))
    return (type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol))


def validate_sequence(config, samples, rows, errors, status):
    failures = []
    cp, dp, world = config['cp'], config['dp'], config['cp'] * config['dp']
    if len(samples) != config['copies'] * dp or not rows:
        return dict(passed=False, failures=['missing input samples or rank/update evidence'],
                    updates=0, workers=world, complete_dataflow_checked=False)
    order = config['order']
    if status not in ('failed', 'completed'):
        return dict(passed=False, failures=['infrastructure status cannot establish candidate behavior'],
                    updates=0, workers=world, complete_dataflow_checked=False)
    expected_updates = []
    for update in range(config['updates']):
        batches = assignment(config, update)
        rank_expected = {}
        scalar = {sid: scalar_sample(sample, cp) for sid, sample in enumerate(samples)}
        for d in range(dp):
            for c in range(cp):
                rank = order[d * cp + c]
                ids = batches[d]
                raw_parts = [scalar[sid]['advantages'][i] for sid in ids for i in scalar[sid]['indices'][c]]
                mask_parts = [samples[sid]['masks'][i] for sid in ids for i in scalar[sid]['indices'][c]]
                members = sorted(order[other_d * cp + c] for other_d in range(dp))
                rank_expected[rank] = dict(
                    samples=ids,
                    local_values=[[samples[sid]['values'][i] for i in scalar[sid]['indices'][c]] for sid in ids],
                    raw_advantages=[[scalar[sid]['advantages'][i] for i in scalar[sid]['indices'][c]] for sid in ids],
                    returns=[[scalar[sid]['returns'][i] for i in scalar[sid]['indices'][c]] for sid in ids],
                    whitening_input=raw_parts, whitening_mask=mask_parts, whitening_members=members,
                    should_enter_whiten=(bool(raw_parts) if status == 'failed' else True))
        for c in range(cp):
            ranks = [order[d * cp + c] for d in range(dp)]
            selected = [v for rank in ranks for v, mask in zip(
                rank_expected[rank]['whitening_input'], rank_expected[rank]['whitening_mask']) if mask]
            if not selected:
                mean, variance = 0.0, 0.0
            else:
                mean = sum(selected) / len(selected)
                variance = sum((value - mean) ** 2 for value in selected) / (len(selected) - 1) if len(selected) > 1 else 0.0
            denom = math.sqrt(variance + 1e-8)
            for rank in ranks:
                exp = rank_expected[rank]
                normalized = []
                for chunk in exp['raw_advantages']:
                    normalized.append([(value - mean) / denom for value in chunk])
                exp['normalized'] = normalized
        expected_updates.append(rank_expected)

        # Native aggregate_train_losses reduces per-sample means across CP×DP.
        # Reconstruct each original sample from its CP-local coordinates first.
        metric_adv, metric_ret, sample_count = 0.0, 0.0, 0
        for d in range(dp):
            ids = batches[d]
            for position, sid in enumerate(ids):
                mask = samples[sid]['masks']
                denominator = max(sum(mask), 1)
                for c in range(cp):
                    rank = order[d * cp + c]
                    expected_row = rank_expected[rank]
                    metric_adv += sum(
                        value * value * mask[index]
                        for value, index in zip(expected_row['normalized'][position],
                                                scalar[sid]['indices'][c], strict=True)
                    ) / denominator
                    metric_ret += sum(
                        value * value * mask[index]
                        for value, index in zip(expected_row['returns'][position],
                                                scalar[sid]['indices'][c], strict=True)
                    ) / denominator
                sample_count += 1
        for rank in rank_expected:
            rank_expected[rank]['report_metrics'] = {
                'adv_sample_mean_square': metric_adv / sample_count,
                'return_sample_mean_square': metric_ret / sample_count,
            }

    if isinstance(rows, dict):
        rows = {int(k): v for k, v in rows.items()}
        if set(rows) != set(range(len(rows))):
            return dict(passed=False, failures=['missing contiguous update evidence'],
                        updates=len(rows), workers=world, complete_dataflow_checked=False)
        rows = [rows[i] for i in range(len(rows))]
    if not 0 < len(rows) <= config['updates']:
        return dict(passed=False, failures=['invalid update evidence count'],
                    updates=len(rows), workers=world, complete_dataflow_checked=False)
    rows = [{str(rank): value for rank, value in update_rows.items()} for update_rows in rows]
    if any(set(row) != set(map(str, range(world))) for row in rows):
        return dict(passed=False, failures=['missing rank evidence'],
                    updates=len(rows), workers=world, complete_dataflow_checked=False)
    for update, rank_rows in enumerate(rows):
        expected = expected_updates[update]
        if set(map(int, rank_rows)) != set(range(world)):
            failures.append(f'update {update}: missing rank records')
            continue
        for rank_string, actual in rank_rows.items():
            rank = int(rank_string)
            exp = expected[rank]
            if actual.get('sample_ids') != exp['samples']:
                failures.append(f'update {update} rank {rank}: assignment mismatch')
            if not _close(actual.get('local_values'), exp['local_values']):
                failures.append(f'update {update} rank {rank}: CP slice mismatch')
            if actual.get('whitening_called') != exp['should_enter_whiten']:
                failures.append(f'update {update} rank {rank}: whitening participation mismatch')
            if exp['should_enter_whiten'] and actual.get('whitening_members') is None:
                failures.append(f'update {update} rank {rank}: missing native whitening group')
            if exp['should_enter_whiten'] and (actual.get('whitening_values') is None
                                               or actual.get('whitening_mask') is None):
                failures.append(f'update {update} rank {rank}: missing whitening inputs')
            if actual.get('whitening_members') is not None and actual['whitening_members'] != exp['whitening_members']:
                failures.append(f'update {update} rank {rank}: wrong DP membership')
            if actual.get('whitening_values') is not None and not _close(actual['whitening_values'], exp['whitening_input']):
                failures.append(f'update {update} rank {rank}: raw GAE values mismatch')
            if not _close(actual.get('raw_advantages'), exp['raw_advantages']):
                failures.append(f'update {update} rank {rank}: pre-normalized GAE mismatch')
            if not _close(actual.get('raw_returns'), exp['returns']):
                failures.append(f'update {update} rank {rank}: raw returns mismatch')
            if actual.get('whitening_mask') is not None and not _close(actual['whitening_mask'], exp['whitening_mask']):
                failures.append(f'update {update} rank {rank}: local mask mismatch')
            if actual.get('whitening_output') is not None:
                flat = [value for chunk in exp['normalized'] for value in chunk]
                if not _close(actual['whitening_output'], flat):
                    failures.append(f'update {update} rank {rank}: normalized whitening mismatch')
            if actual.get('status') in ('computed', 'aggregate_error'):
                if not _close(actual.get('advantages'), exp['normalized']):
                    failures.append(f'update {update} rank {rank}: normalized advantages mismatch')
                if not _close(actual.get('returns'), exp['returns']):
                    failures.append(f'update {update} rank {rank}: returns mismatch')
            if (config['reporting'] and actual.get('status') == 'computed'
                    and actual.get('report_metrics') is None):
                failures.append(f'update {update} rank {rank}: missing completed report')
            if actual.get('report_metrics') is not None:
                for key, expected_value in exp['report_metrics'].items():
                    if not _close(actual['report_metrics'].get(key), expected_value):
                        failures.append(f'update {update} rank {rank}: aggregated {key} mismatch')

    if status == 'failed':
        failed_update = len(rows) - 1
        for update in range(failed_update):
            if any(rows[update][str(rank)].get('status') != 'computed'
                   for rank in range(world)):
                failures.append(f'completed prefix update {update} is inconsistent')
        compute_errors = [e for e in errors if e['update'] == failed_update and e['operation'] == 'compute_advantages_and_returns']
        metric_errors = [e for e in errors if e['update'] == failed_update and e['operation'] == 'aggregate_train_losses']
        if not compute_errors:
            failures.append('missing later compute-side collective failure')
        elif not any('timed out' in e.get('message', '').lower() for e in compute_errors):
            failures.append('compute-side peer error is not a target collective timeout')
        if config.get('reporting', True) and not metric_errors:
            failures.append('missing downstream metric timeout symptom')
        elif metric_errors and not any('timed out' in e.get('message', '').lower() for e in metric_errors):
            failures.append('downstream symptom is not a timed collective failure')
        expected_empty = [rank for rank, exp in expected_updates[failed_update].items()
                          if not exp['should_enter_whiten']]
        for rank in expected_empty:
            if rows[failed_update][str(rank)].get('whitening_called'):
                failures.append(f'update {failed_update} rank {rank}: empty shard unexpectedly entered whitening')
            d, c = divmod(order.index(rank), cp)
            peer = order[(1 - d) * cp + c]
            peer_row = rows[failed_update].get(str(peer), {})
            if not peer_row.get('whitening_called'):
                failures.append(f'update {failed_update} rank {peer}: nonempty peer did not enter whitening')
            if not peer_row.get('whitening_error_type'):
                failures.append(f'update {failed_update} rank {peer}: whitening did not retain failure evidence')
            if not any(e['update'] == failed_update and e['rank'] == peer
                       and e['operation'] == 'compute_advantages_and_returns' for e in errors):
                failures.append(f'update {failed_update} rank {peer}: missing blocked compute error')
        if config.get('reporting', True) and not any(
                e['update'] == failed_update and e['operation'] == 'aggregate_train_losses'
                for e in errors):
            failures.append('missing downstream reporting-stage symptom')
    if status == 'completed':
        if errors or len(rows) != config['updates']:
            failures.append('successful run is missing updates or contains errors')
        if any(row.get('status') != 'computed' for update in rows for row in update.values()):
            failures.append('successful run contains unfinished workers')
    return dict(passed=not failures, failures=failures,
                updates=len(rows), workers=world,
                complete_dataflow_checked=True)
