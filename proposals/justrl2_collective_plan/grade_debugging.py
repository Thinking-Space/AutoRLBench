#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional reviewer audit of coherent before/control/after replay artifacts.

This is an evidence-consistency gate, not tamper-proof execution attestation.
Its artifact schema is not a solver submission requirement.
Use verify_runtime.py separately with a trusted runner for fresh repair checks.
"""
import argparse
import json
from pathlib import Path

import replay_probe
from verify_runtime import assess


def _experiment_key(config):
    return json.dumps({key: config[key] for key in
                       ('cp', 'dp', 'logical_to_global', 'steps', 'mask_mode', 'seed')}, sort_keys=True)


def validate(manifest, original, repaired):
    result = {'passed': False, 'errors': [], 'scope': 'artifact consistency; requires separate trusted replay and report review'}
    try:
        manifest = Path(manifest).resolve()
        payload = json.loads(manifest.read_text())
        report = Path(payload['report'])
        if not report.is_absolute():
            report = manifest.parent / report
        if not report.is_file() or not report.read_text().strip():
            raise ValueError('A nonempty debugging report is required')
        hashes = {'before': replay_probe._source_functions(Path(original))[1],
                  'after': replay_probe._source_functions(Path(repaired))[1]}
        cases = {'before': {}, 'after': {}}
        all_paths = set()
        for phase in ('before', 'after'):
            paths = payload.get(phase)
            if not isinstance(paths, list) or not paths or len(paths) > 32:
                raise ValueError('Supply 1..32 experiment directories per phase')
            for value in paths:
                directory = Path(value)
                if not directory.is_absolute():
                    directory = manifest.parent / directory
                directory = directory.resolve()
                if directory in all_paths:
                    raise ValueError('The same artifact directory cannot count as multiple experiments')
                all_paths.add(directory)
                summary = json.loads((directory / 'summary.json').read_text())
                config = summary['config']
                validated = replay_probe._configuration(config['cp'], config['dp'], config['logical_to_global'],
                                                        config['steps'], config['mask_mode'], config['seed'], config['trace'])
                if config['world_size'] != validated['world_size']:
                    raise ValueError('World size does not match CP times DP')
                if not config['trace'] or summary['source_sha256'] != hashes[phase]:
                    raise ValueError('Evidence must contain real traces bound to the declared source version')
                if summary.get('source_sha256_after') != hashes[phase]:
                    raise ValueError('Source changed or source-after hash is absent')
                key = _experiment_key(config)
                if key in cases[phase]:
                    raise ValueError('Repeated configurations do not supply independent controls')
                assessment = assess(directory)
                if assessment['inconsistencies']:
                    raise ValueError('Inconsistent inputs or completed-prefix evidence: ' + assessment['inconsistencies'][0])
                cases[phase][key] = assessment
        failures = [c for c in cases['before'].values() if not c['completed'] and c['genuine_misroute_timeouts']]
        phases = {c['misroutes'][0]['phase'] for c in failures if c['misroutes']}
        if len(failures) < 2 or len(phases) < 2:
            raise ValueError('Need two history-dependent failures with distinct first misrouting phases')
        controls = [c for c in cases['before'].values() if c['numerical_passed']]
        if len(controls) < 2:
            raise ValueError('Need at least two distinct passing controls with numerical evidence')
        if len(failures) + len(controls) != len(cases['before']):
            raise ValueError('Unexplained or infrastructure errors cannot be debugging evidence')
        if not set(cases['before']) < set(cases['after']):
            raise ValueError('Repeat every pre-repair experiment and add an unseen configuration after repair')
        if not all(c['numerical_passed'] for c in cases['after'].values()):
            raise ValueError('Every post-repair experiment must pass routing and independent numerics')
        result.update(passed=True, failures=len(failures), passing_controls=len(controls),
                      repaired_experiments=len(cases['after']), first_failure_phases=sorted(phases))
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repaired', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.manifest, args.original, args.repaired)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
