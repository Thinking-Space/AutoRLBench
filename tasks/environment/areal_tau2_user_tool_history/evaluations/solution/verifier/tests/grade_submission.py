"""Grade the fresh telecom rollout plus the completion adapter's compatibility."""

import asyncio
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path


def main():
    verifier = Path(__file__).resolve().parents[1]
    spec = json.loads((verifier / 'resources/grader_spec.json').read_text())
    workspace = Path(os.environ.get('AUTORLBENCH_WORKSPACE', '/workspace'))
    results = Path(os.environ.get('AUTORLBENCH_RESULTS', '/autorlbench/results'))
    results.mkdir(parents=True, exist_ok=True)
    passed = False
    detail = {}
    try:
        if not (workspace / 'report.md').is_file() or not (workspace / 'report.md').read_text().strip():
            raise ValueError('A nonempty report.md is required')
        log_root = results / 'experiment_logs'
        summary = json.loads((log_root / 'summary.json').read_text())
        metrics = json.loads((log_root / 'experiment_metrics.json').read_text())
        expected = int(spec['final_step'])
        if summary['episodes'] != expected or int(metrics['step']) != expected:
            raise ValueError('The declared rollout horizon must complete')
        if summary['valid_episodes'] < int(spec['minimum_valid_episodes']):
            raise ValueError('The minimum error-free trajectory count is required')
        before = summary['user_tool_messages_before'] + summary['user_tool_calls_before']
        after = summary['user_tool_messages_after'] + summary['user_tool_calls_after']
        if before <= 0 or after < 0 or after > before:
            raise ValueError('The user tool history must be exercised with consistent counters')
        ratio = Decimal(after) / Decimal(before)
        logged = Decimal(str(metrics[spec['metric_key']]))
        if abs(ratio - logged) > Decimal('0.000000001'):
            raise ValueError('The retention metric must match the fresh request counters')
        loader = importlib.util.spec_from_file_location('contract_checks', verifier / 'tests/check_messages.py')
        checks = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(checks)
        compatibility = asyncio.run(checks.check(workspace / 'areal/examples/tau2/agent.py'))
        detail = {'retention': float(ratio), 'episodes': summary['episodes'],
                  'valid_episodes': summary['valid_episodes'], 'mean_reward': summary['mean_reward'],
                  'compatibility': compatibility}
        passed = (Decimal(spec['required_final_score']) <= ratio <=
                  Decimal(spec['required_final_score_max']) and compatibility['passed'])
    except Exception as error:
        detail['error'] = f'{type(error).__name__}: {error}'
    result = {'passed': passed, 'score': 1.0 if passed else 0.0, 'metrics': detail}
    for filename in ('experiment_metrics.json', 'metrics.json', 'result.json'):
        (results / filename).write_text(json.dumps(result if filename == 'result.json' else detail,
                                                   indent=2))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
