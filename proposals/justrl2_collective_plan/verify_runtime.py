#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only fresh replay and independent numerical acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import replay_probe
from numerical_reference import numeric_equal, reference_step
from plan_identity import check as check_identity

CASES = {
    'long': dict(cp=2, dp=2, steps=[11]),
    'cold_short': dict(cp=2, dp=2, steps=[3]),
    'warm_short': dict(cp=2, dp=2, steps=[11, 3]),
    'reverse_order': dict(cp=2, dp=2, steps=[3, 11]),
    'cp1': dict(cp=1, dp=2, steps=[11, 3]),
    'cp4_long': dict(cp=4, dp=2, steps=[11, 17]),
    'unequal_dp3': dict(cp=2, dp=3, steps=[11, 3]),
    'permuted': dict(cp=2, dp=2, order=[0, 2, 1, 3], steps=[11, 3]),
    'sparse_unseen': dict(cp=4, dp=2, order=[0, 2, 4, 6, 1, 3, 5, 7],
                          steps=[7, 11, 17], mask_mode='sparse', seed=90210),
}


def trace_coverage(traces, step, cp, expected_cp, expected_dp):
    """Completed updates need real matched native calls, not request-only logs."""
    events = [e for e in traces if e['step'] == step]
    requests = [e for e in events if e['operation'] == 'reduce_sum' and e['event'] == 'enter']
    phases = [e['phase'] for e in requests]
    if phases != (['gae', 'gae', 'whiten'] if cp > 1 else ['whiten']):
        return False
    if len(events) != 4 * len(requests):
        return False
    for request in requests:
        call = [e for e in events if e['call'] == request['call']]
        if [(e['operation'], e['event']) for e in call] != [
            ('reduce_sum', 'enter'), ('all_reduce', 'enter'),
            ('all_reduce', 'return'), ('reduce_sum', 'return')]:
            return False
        expected = expected_cp if request['phase'] == 'gae' else expected_dp
        if any(e['requested_members'] != expected for e in call):
            return False
        for event in call[1:3]:
            if (event['actual_members'] != expected or event['shape'] != request['shape']
                    or event['numel'] != request['numel']):
                return False
    return True


def assess(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    config = summary['config']
    validated = replay_probe._configuration(config['cp'], config['dp'], config['logical_to_global'],
                                             config['steps'], config['mask_mode'], config['seed'], config['trace'])
    if config['world_size'] != validated['world_size']:
        raise ValueError('World size does not match CP times DP')
    result = dict(completed=summary['status'] == 'completed', numerical_passed=False,
                  misroutes=[], genuine_misroute_timeouts=[], failures=[], inconsistencies=[])
    if summary['status'] == 'infrastructure_error' or summary['outer_timeout']:
        result['failures'].append('Infrastructure failure is not candidate reproduction')
        return result
    rows, trace_rows = {}, {}
    for rank in range(config['world_size']):
        row = json.loads((directory / f'rank{rank}.json').read_text())
        if row['rank'] != rank or not row['done']:
            raise ValueError('Missing completed rank')
        rows[rank] = row
        traces = [json.loads(s) for s in (directory / f'rank{rank}.trace.jsonl').read_text().splitlines()]
        trace_rows[rank] = traces
        if not traces or [e['sequence'] for e in traces] != list(range(len(traces))):
            raise ValueError('Missing or unordered actual communication trace')
        for event in traces:
            if event['operation'] != 'all_reduce' or event['requested_members'] == event['actual_members']:
                continue
            witness = {k: event[k] for k in ('rank', 'step', 'phase', 'call', 'shape', 'numel',
                                            'requested_members', 'actual_members')}
            if event['event'] == 'enter':
                result['misroutes'].append(witness)
            if (event['event'] == 'error' and event.get('elapsed', 0) >= 0.8 * replay_probe.COLLECTIVE_TIMEOUT
                    and 'gloo' in event.get('error', '').lower() and 'timed out' in event['error'].lower()):
                result['genuine_misroute_timeouts'].append(witness)
    cp, dp, order = config['cp'], config['dp'], config['logical_to_global']
    for step, length in enumerate(config['steps']):
        canonical = [replay_probe._samples(length, cp, d, config['seed'], step, config['mask_mode']) for d in range(dp)]
        samples = [[dict(total_length=s['total'], response_length=s['length'], values=s['values'],
                         loss_mask=s['masks'], reward=s['reward']) for s in row] for row in canonical]
        expected = reference_step(samples, cp, order)
        for rank, row in rows.items():
            if step >= len(row['steps']):
                result['failures'].append(f'rank {rank} did not execute update {step}')
                continue
            actual = row['steps'][step]
            d = order.index(rank) // cp
            if actual['samples'] != canonical[d]:
                error = f'rank {rank} update {step}: input mismatch'
                result['failures'].append(error)
                result['inconsistencies'].append(error)
            if actual['status'] != 'completed':
                result['failures'].append(f'rank {rank} update {step}: candidate error')
                continue
            expected_cp = order[d * cp:(d + 1) * cp]
            if not trace_coverage(trace_rows[rank], step, cp, expected_cp, expected[rank]['group']):
                error = f'rank {rank} update {step}: missing or inconsistent actual collective coverage'
                result['failures'].append(error)
                result['inconsistencies'].append(error)
            for key, ref_key in [('sliced_values', 'sliced_values'), ('raw_advantages', 'raw'),
                                 ('raw_returns', 'returns'), ('advantages', 'advantages'), ('returns', 'returns')]:
                if not numeric_equal(actual.get(key), expected[rank][ref_key]):
                    error = f'rank {rank} update {step}: {key} differs from scalar reference'
                    result['failures'].append(error)
                    result['inconsistencies'].append(error)
    result['numerical_passed'] = result['completed'] and not result['failures'] and not result['misroutes']
    return result


def verify(source, output, cases=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = list(CASES) if cases is None else list(cases)
    if not selected or set(selected) - CASES.keys() or len(selected) != len(set(selected)):
        raise ValueError('Select nonempty unique named cases')
    result = dict(passed=False, source_sha256=replay_probe._source_functions(source)[1], cases={},
                  scope='Fresh CPU/Gloo native functions and constructed plan; no GPU/model validation')
    for name in selected:
        replay_probe.run_probe(source, output / name, **CASES[name], trace=True)
        result['cases'][name] = assess(output / name)
    result['identity'] = check_identity(source / replay_probe.PLAN, output / 'identity')
    result['passed'] = all(c['numerical_passed'] for c in result['cases'].values()) and result['identity']['passed']
    if replay_probe._source_functions(source)[1] != result['source_sha256']:
        result.update(passed=False, source_changed=True)
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source, args.output)
    print(json.dumps({'passed': result['passed'], 'cases': {k: v['numerical_passed'] for k, v in result['cases'].items()}}))
    raise SystemExit(0 if result['passed'] else 1)
