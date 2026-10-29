"""Grade the completed agent training run without executing workspace code."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    verifier = Path(os.environ['AUTORLBENCH_VERIFIER'])
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    results = Path(os.environ['AUTORLBENCH_RESULTS']).resolve()
    results.mkdir(parents=True, exist_ok=True)
    spec = json.loads((verifier/'resources/grader_spec.json').read_text())
    report = {'grading_protocol': 'agent-training-artifacts-v1', 'passed': False, 'score': 0.0, 'metrics': {}}
    try:
        assert spec.get('calibration_status') == 'validated', 'GSM8K calibration is pending; checkpoint testing is deferred'
        for name, digest in spec['protected_sha256'].items():
            path = workspace/name
            assert path.is_file() and not path.is_symlink() and sha256(path) == digest, 'fixed data or historical evidence changed: '+name
        for asset in spec['model_assets']:
            path = workspace/'model'/asset['path']
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == asset['size'] and sha256(path) == asset['sha256'], 'model asset changed: '+asset['path']
        output = results/'experiment_logs'
        assert output.is_dir() and not output.is_symlink(), 'missing agent training directory: experiment_logs'
        completion = json.loads((output/'completed.json').read_text())
        stages = ['before', 'after'] if spec['kind'] == 'configuration' else ['full']
        assert completion.get('completed') is True and completion.get('stages') == stages, 'agent training did not complete the required stages'
        report['metrics']['training_dir'] = str(output)
        from check_run import evaluate, final_evaluation
        final, count = final_evaluation(output, workspace/'data', spec['final_step'])
        report['metrics'].update(final_accuracy=final, final_step=spec['final_step'], final_held_out_responses=count)
        assert 0.0 <= final <= 1.0, 'invalid final held-out accuracy'
        assert final > spec['final_accuracy_threshold_exclusive'], 'final held-out accuracy must be strictly greater than 45%'
        assert round(final*count) >= spec['minimum_correct_final_answers'], 'final held-out evaluation has no independently correct answer'
        report['metrics'].update(evaluate(output, spec['kind'], spec['final_step'], data_root=workspace/'data'))
        value = report['metrics'][spec['metric_key']]
        assert spec['required_final_score'] <= value <= spec['required_final_score_max'], 'runtime metric is outside required bounds'
        report.update(passed=True, score=1.0)
    except Exception as error:
        report['metrics']['error'] = str(error)
    (results/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
