"""Local orchestration regressions; fake Docker does NOT test container isolation."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib

import pytest

from tests.justrl2_review.fixtures import (
    TASK, REFERENCE_LOADER, clean_env, inventory, load, read, write,
)

EVALUATOR = Path('provenance/evaluate_submission.py')


def options(command, flag):
    return [command[index + 1] for index, value in enumerate(command) if value == flag]


def mounts(command):
    result = {}
    for item in options(command, '--mount'):
        fields = dict(part.split('=', 1) if '=' in part else (part, True)
                      for part in item.split(','))
        result[fields['target']] = fields
    return result


def submission(root, *, healthy=False, repair=True, task=TASK):
    root.mkdir()
    source = REFERENCE_LOADER if repair else task / 'workspace/loader.py'
    if healthy:
        source = task / 'workspace/loader.py'
    shutil.copyfile(source, root / 'loader.py')
    (root / 'report.md').write_text(
        f'Verdict: {"healthy" if healthy else "unhealthy"}\n\nReviewed current code.\n')
    # These files must never be mounted or copied into either container.
    (root / 'run.sh').write_text('exit 99\n')
    write(root / 'result.json', {'score': 1, 'passed': True})
    (root / 'sitecustomize.py').write_text('raise RuntimeError("submitted startup code")\n')
    return root


class HostScripts:
    """Interpret Docker bind paths, then run the actual trusted scripts locally."""
    def __init__(self):
        self.commands = []
        self.private_roots = []

    def __call__(self, command, **kwargs):
        assert kwargs.get('shell', False) is False
        assert kwargs['timeout'] <= 180
        self.commands.append(command)
        bound = mounts(command)
        workspace = Path(bound['/workspace']['source'])
        self.private_roots.append(workspace.parent)
        bin_dir = workspace.parent / 'host-test-bin'
        bin_dir.mkdir(exist_ok=True)
        wrapper = bin_dir / 'python3'
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -B "$@"\n')
        wrapper.chmod(0o755)
        env = clean_env(PATH=str(bin_dir) + os.pathsep + os.defpath)
        for assignment in options(command, '--env'):
            key, value = assignment.split('=', 1)
            env[key] = bound.get(value, {}).get('source', value)
        if '/autorlbench/environment' in bound:
            assert set(path.name for path in workspace.iterdir()) == {'loader.py', 'report.md'}
            script = Path(bound['/autorlbench/environment']['source']) / 'prepare.sh'
        else:
            script = Path(bound['/verifier']['source']) / 'test.sh'
        return subprocess.run(['/bin/bash', '--noprofile', '--norc', str(script)],
                              env=env, cwd=workspace, capture_output=True, text=True,
                              timeout=45, stdin=subprocess.DEVNULL)


def test_command_plan_uses_separate_pinned_cpu_containers_and_minimal_mounts(tmp_path):
    evaluator = load(TASK / EVALUATOR)
    workspace, prepared, results = [tmp_path / name for name in ('workspace', 'prepared', 'results')]
    plan = evaluator.command_plan(TASK, workspace, prepared, results)
    assert len(plan) == 2
    image = tomllib.loads((TASK / 'task.toml').read_text())['environment']['image']
    assert '@sha256:' in image
    names = []
    for command in plan:
        assert command[:2] == ['docker', 'run']
        assert options(command, '--network') == ['none']
        assert options(command, '--entrypoint') == ['/bin/bash']
        assert options(command, '--user') == [f'{os.getuid()}:{os.getgid()}']
        assert options(command, '--cap-drop') == ['ALL']
        assert options(command, '--security-opt') == ['no-new-privileges']
        assert options(command, '--memory') == ['1g']
        assert options(command, '--memory-swap') == ['1g']
        assert options(command, '--pids-limit') == ['64']
        assert options(command, '--pull') == ['never']
        assert '--read-only' in command and '--rm' in command
        assert '--gpus' not in command and '--privileged' not in command
        assert 'NVIDIA_VISIBLE_DEVICES=void' in options(command, '--env')
        assert 'BASH_ENV=/dev/null' in options(command, '--env')
        assert {'rw', 'nosuid', 'nodev'} <= set(options(command, '--tmpfs')[0].split(':')[1].split(','))
        assert command[command.index(image) + 1:][:2] == ['--noprofile', '--norc']
        names.extend(options(command, '--name'))
    assert len(set(names)) == 2
    first, second = map(mounts, plan)
    assert set(first) == {'/autorlbench/environment', '/workspace', '/autorlbench/prepared'}
    assert set(second) == {'/verifier', '/workspace', '/autorlbench/prepared', '/results'}
    assert first['/autorlbench/environment']['readonly'] is True
    for target in ('/workspace', '/autorlbench/prepared'):
        assert 'readonly' not in first[target]
        assert second[target]['readonly'] is True
        assert first[target]['source'] == second[target]['source']
    assert second['/verifier']['readonly'] is True
    assert 'readonly' not in second['/results']
    assert plan[0][-1] == '/autorlbench/environment/prepare.sh'
    assert plan[1][-1] == '/verifier/test.sh'
    bad_task = tmp_path / 'unpinned-task'
    bad_task.mkdir()
    (bad_task / 'task.toml').write_text('[environment]\nimage = "example:latest"\n')
    with pytest.raises(ValueError, match='digest|sha256'):
        evaluator.command_plan(bad_task, workspace, prepared, results)


def test_staging_is_bounded_nonsymlink_and_existing_output_is_never_touched(tmp_path, capsys):
    evaluator = load(TASK / EVALUATOR)
    source = submission(tmp_path / 'submission with spaces;literal')
    staged = tmp_path / 'stage'
    staged.mkdir()
    evaluator.stage_submission(source, staged)
    assert set(path.name for path in staged.iterdir()) == {'loader.py', 'report.md'}
    assert (staged / 'loader.py').read_bytes() == (source / 'loader.py').read_bytes()
    (source / 'loader.py').write_bytes(b'#' * (65536 + 1))
    with pytest.raises(ValueError, match='64|65536|large'):
        evaluator.stage_submission(source, tmp_path / 'oversized')
    (source / 'loader.py').write_bytes(REFERENCE_LOADER.read_bytes())
    target = tmp_path / 'target-report'
    (source / 'report.md').replace(target)
    (source / 'report.md').symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        evaluator.stage_submission(source, tmp_path / 'linked')
    (source / 'report.md').unlink()
    shutil.copyfile(target, source / 'report.md')
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'sentinel').write_text('untouched')
    calls = []
    assert evaluator.main(['--submission', str(source), '--output', str(output)],
                          executor=lambda *a, **kw: calls.append(a)) == 2
    assert read_stdout(capsys)['status'] == 'error'
    assert (output / 'sentinel').read_text() == 'untouched' and not calls
    link = tmp_path / 'output-link'
    link.symlink_to(tmp_path / 'missing-output', target_is_directory=True)
    assert evaluator.main(['--submission', str(source), '--output', str(link)],
                          executor=lambda *a, **kw: calls.append(a)) == 2
    assert read_stdout(capsys)['status'] == 'error'
    assert link.is_symlink() and not link.exists() and not calls


def read_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_current_and_generated_control_grade_real_scripts_with_private_staging(tmp_path, capsys):
    control = load(TASK / 'provenance/make_control.py').make_control(tmp_path / 'control')
    for task, healthy in ((TASK, False), (control, True)):
        root = tmp_path / ('healthy' if healthy else 'repaired')
        root.mkdir()
        source = submission(root / 'submission', healthy=healthy, task=task)
        before = inventory(source)
        executor = HostScripts()
        output = root / 'results'
        evaluator = load(task / EVALUATOR)
        code = evaluator.main(['--submission', str(source), '--output', str(output)], executor=executor)
        result = read_stdout(capsys)
        assert code == 0 and result['score'] == 1 and result['passed'] is True
        assert result == read(output / 'result.json') == read(output / 'metrics.json')
        assert set(path.name for path in output.iterdir()) == {'result.json', 'metrics.json'}
        assert result['metrics']['checked_records'] == 128
        assert result['assessment']['expected'] == ('healthy' if healthy else 'unhealthy')
        assert len(executor.commands) == 2
        assert all(not private.exists() for private in executor.private_roots)
        assert inventory(source) == before
        assert all(str(source) not in str(command) for command in executor.commands)


def test_valid_grade_failure_is_fresh_and_exits_one(tmp_path, capsys):
    evaluator = load(TASK / EVALUATOR)
    source = submission(tmp_path / 'submission', repair=False)
    executor = HostScripts()
    output = tmp_path / 'results'
    assert evaluator.main(['--submission', str(source), '--output', str(output)], executor=executor) == 1
    result = read_stdout(capsys)
    assert result['score'] == 0 and result['passed'] is False
    assert result['metrics']['score'] is False
    assert result == read(output / 'result.json') == read(output / 'metrics.json')
    assert result['assessment']['verdict_correct'] is True
    assert len(executor.commands) == 2
    assert all(not private.exists() for private in executor.private_roots)


@pytest.mark.parametrize('failure', ['prepare', 'docker', 'missing', 'schema', 'status'])
def test_infrastructure_failures_cannot_publish_forged_or_malformed_success(tmp_path, capsys, failure):
    evaluator = load(TASK / EVALUATOR)
    source = submission(tmp_path / 'submission')
    calls = []
    private_roots = []

    def broken(command, **kwargs):
        calls.append(command)
        bound = mounts(command)
        private_roots.append(Path(bound['/workspace']['source']).parent)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 125 if failure == 'prepare' else 0,
                                               stdout='prepare log\n', stderr='diagnostic\n')
        result = {'score': 1, 'passed': True, 'assessment': {}, 'metrics': {}}
        results = Path(bound['/results']['source'])
        if failure != 'missing':
            for name in ('result.json', 'metrics.json'):
                write(results / name, result)
        status = 125 if failure == 'docker' else 1 if failure == 'status' else 0
        return subprocess.CompletedProcess(command, status, json.dumps(result), 'diagnostic\n')

    output = tmp_path / failure
    assert evaluator.main(['--submission', str(source), '--output', str(output)], executor=broken) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)['status'] == 'error'
    assert 'diagnostic' in captured.err
    assert not (output / 'result.json').exists()
    assert not (output / 'metrics.json').exists()
    assert len(calls) == (1 if failure == 'prepare' else 2)
    assert all(not private.exists() for private in private_roots)
