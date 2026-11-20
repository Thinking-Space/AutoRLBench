"""The healthy control retains identical neutral instructions and data."""
from pathlib import Path
import sys

import pytest

from tests.justrl2_review.fixtures import (
    TASK, ReviewRun, assert_reads_pass, execute, inventory, load, read,
)


def test_generated_control_validates_and_accepts_unchanged_code_via_cpu_run_spec(tmp_path):
    from autorlbench.tasks.loader import load_task_package

    generator = load(TASK / 'provenance/make_control.py')
    control = generator.make_control(tmp_path / 'control')
    package = load_task_package(control)
    assert package.report.ok, package.report.to_dict()
    assert package.metadata['outcome']['failure_type'] == 'healthy_control'
    assert package.metadata['environment']['gpus'] == 0
    assert package.metadata['security']['requires_isolated_stages'] is True
    for tree in ('environment', 'workspace/logs'):
        assert inventory(control / tree) == inventory(TASK / tree)
    for level in ('issue', 'solution'):
        relative = f'evaluations/{level}/instruction.md'
        assert (control / relative).read_bytes() == (TASK / relative).read_bytes()
    assert read(control / 'evaluations/solution/verifier/resources/expected_verdict.json') == {
        'schema_version': 1, 'expected_verdict': 'healthy'}
    run = ReviewRun(tmp_path / 'review', 'healthy', task=control)
    run.submit('healthy')
    spec = read(control / 'run_spec.json')
    assert 'check_reads.py' in spec['command']
    assert all(flag not in spec['command'] for flag in ('--native', '--gpus', 'checkpoint'))
    # Execute the shipped task command with only the system Python available.
    env = dict(run.env)
    for key, value in spec.get('env', {}).items():
        for name, replacement in run.env.items():
            value = value.replace('${' + name + '}', replacement).replace('$' + name, replacement)
        env[key] = value
    env['PATH'] = str(Path(sys.executable).parent) + ':/usr/bin:/bin'
    process = execute(['/bin/bash', '-c', spec['command']], env=env, cwd=run.workspace)
    assert process.returncode == 0, process.stdout + process.stderr
    result = run.run()
    assert result['score'] == 1
    assert_reads_pass(result['metrics'])
    assert run.loader.read_bytes() == run.initial_loader
    with pytest.raises(FileExistsError):
        generator.make_control(control)
