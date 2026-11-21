#!/usr/bin/env python3
"""Grade submitted files in two isolated Docker containers (Python 3.11+).

Usage: python3 provenance/evaluate_submission.py --submission WORKSPACE --output NEW_RESULTS
The digest-pinned task image must already be available to the Docker daemon.
This evaluates artifacts; it does not launch an agent or execute submitted code.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import tomllib

MAX_ARTIFACT = 65536
MAX_RESULT = 4 * 1024 * 1024


def _read_regular(name, limit, *, directory_fd=None):
    """Open the same regular inode we validate, without following a leaf link."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Expected a regular nonsymlink file: ' + str(name))
        if info.st_size > limit:
            raise ValueError(f'File exceeds {limit} bytes: {name}')
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f'File exceeds {limit} bytes: {name}')
        return raw


def stage_submission(source, destination):
    """Copy only the two permitted bounded artifacts into a private directory."""
    descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        artifacts = {name: _read_regular(name, MAX_ARTIFACT, directory_fd=descriptor)
                     for name in ('loader.py', 'report.md')}
    finally:
        os.close(descriptor)
    destination = Path(destination)
    destination.mkdir(mode=0o700, exist_ok=True)
    if destination.is_symlink() or any(destination.iterdir()):
        raise ValueError('Staging destination must be an empty real directory')
    for name, raw in artifacts.items():
        path = destination / name
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(0o600)


def _mount(source, target, readonly=False):
    source = str(Path(source).resolve())
    if any(char in source for char in (',', '\n', '\r')):
        raise ValueError('Docker bind paths cannot contain commas or newlines')
    return f'type=bind,source={source},target={target}' + (',readonly' if readonly else '')


def command_plan(task, workspace, prepared, results):
    """Build fixed prepare/verifier argv; no submission supplies commands."""
    task = Path(task).resolve()
    image = tomllib.loads((task / 'task.toml').read_text())['environment']['image']
    if not isinstance(image, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}', image):
        raise ValueError('Task image must be pinned to a sha256 digest')
    token = secrets.token_hex(8)
    common = [
        'docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', f'{os.getuid()}:{os.getgid()}', '--memory', '1g',
        '--memory-swap', '1g', '--pids-limit', '64', '--cpus', '1',
        '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777',
        '--no-healthcheck', '--entrypoint', '/bin/bash', '--workdir', '/workspace',
        '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'BASH_ENV=/dev/null',
        '--env', 'ENV=/dev/null', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'AUTORLBENCH_WORKSPACE=/workspace',
        '--env', 'AUTORLBENCH_PREPARED=/autorlbench/prepared',
    ]
    prepare = common + [
        '--name', 'justrl2-prepare-' + token,
        '--env', 'AUTORLBENCH_ENVIRONMENT=/autorlbench/environment',
        '--mount', _mount(task / 'environment', '/autorlbench/environment', True),
        '--mount', _mount(workspace, '/workspace'),
        '--mount', _mount(prepared, '/autorlbench/prepared'),
        image, '--noprofile', '--norc', '/autorlbench/environment/prepare.sh',
    ]
    verify = common + [
        '--name', 'justrl2-verify-' + token,
        '--env', 'AUTORLBENCH_RESULTS=/results',
        '--mount', _mount(workspace, '/workspace', True),
        '--mount', _mount(prepared, '/autorlbench/prepared', True),
        '--mount', _mount(task / 'evaluations/solution/verifier', '/verifier', True),
        '--mount', _mount(results, '/results'),
        image, '--noprofile', '--norc', '/verifier/test.sh',
    ]
    return [prepare, verify]


def _log(value):
    if value:
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='replace')
        print(value, end='' if value.endswith('\n') else '\n', file=sys.stderr, flush=True)


def _run(command, executor):
    try:
        process = executor(command, stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=180, check=False)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        _log(getattr(error, 'stdout', None))
        _log(getattr(error, 'stderr', None))
        # Killing the client on timeout does not stop its container.
        name = command[command.index('--name') + 1]
        try:
            cleanup = executor(['docker', 'rm', '--force', name], capture_output=True,
                               text=True, timeout=15, check=False, stdin=subprocess.DEVNULL)
            _log(cleanup.stderr)
        except (OSError, subprocess.SubprocessError) as cleanup_error:
            _log('Container cleanup failed: ' + str(cleanup_error))
        raise RuntimeError('Container interrupted or exceeded 180 seconds: ' + name) from error
    _log(process.stdout)
    _log(process.stderr)
    return process


def _json(raw):
    def reject_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(raw, parse_constant=reject_constant)


def _validate_result(result, returncode):
    def require(condition):
        if not condition:
            raise ValueError('Invalid verifier result schema or inconsistent exit status')

    require(isinstance(result, dict))
    require(type(result.get('passed')) is bool and type(result.get('score')) is int)
    require(result['score'] in (0, 1) and result['score'] == int(result['passed']))
    require(returncode == (0 if result['passed'] else 1))
    assessment, metrics = result.get('assessment'), result.get('metrics')
    require(isinstance(assessment, dict) and isinstance(metrics, dict))
    for value in (assessment, metrics):
        require(isinstance(value.get('errors'), list)
                and all(isinstance(item, str) for item in value['errors']))
    require(type(assessment.get('report_valid')) is bool)
    require(type(assessment.get('verdict_correct')) is bool)
    require(assessment.get('expected') in ('healthy', 'unhealthy'))
    require(assessment.get('reported') in (None, 'healthy', 'unhealthy'))
    require(type(metrics.get('score')) is bool)
    require(type(metrics.get('checked_records')) is int and 0 <= metrics['checked_records'] <= 128)
    require(isinstance(metrics.get('variants'), list) and isinstance(metrics.get('probes'), list))
    names = []
    total = 0
    for variant in metrics['variants']:
        require(isinstance(variant, dict))
        require(variant.get('name') in ('original', 'relocated') and type(variant.get('passed')) is bool)
        names.append(variant['name'])
        for key in ('readable_count', 'easy', 'hard'):
            require(type(variant.get(key)) is int and 0 <= variant[key] <= 64)
        require(variant['easy'] + variant['hard'] == variant['readable_count'])
        total += variant['readable_count']
    require(len(set(names)) == len(names) and total == metrics['checked_records'])
    for probe in metrics['probes']:
        require(isinstance(probe, dict) and isinstance(probe.get('name'), str)
                and type(probe.get('passed')) is bool)
    require(result['passed'] == (metrics['score'] and assessment['report_valid']
                                 and assessment['verdict_correct']))
    if metrics['score']:
        require(set(names) == {'original', 'relocated'} and not metrics['errors'])
        require(all(item['passed'] and (item['readable_count'], item['easy'], item['hard']) == (64, 32, 32)
                    for item in metrics['variants']))
        require(len(metrics['probes']) == 6 and all(item['passed'] for item in metrics['probes']))
        require({item['name'] for item in metrics['probes']} == {
            'unknown', 'content', 'unknown_existing', 'content_existing',
            'missing_current_file', 'ambiguous_catalog'})


def _write_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None, *, executor=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    executor = subprocess.run if executor is None else executor
    output = None
    try:
        # mkdir atomically refuses existing directories, files and dangling links.
        candidate = args.output.expanduser().absolute()
        candidate.mkdir(mode=0o700)
        output = candidate
        task = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='.justrl2-evaluation-', dir=output.parent) as directory:
            scratch = Path(directory)
            workspace, prepared, results = (scratch / name for name in ('workspace', 'prepared', 'results'))
            stage_submission(args.submission.expanduser(), workspace)
            prepared.mkdir(mode=0o700)
            results.mkdir(mode=0o700)
            prepare_command, verify_command = command_plan(task, workspace, prepared, results)
            process = _run(prepare_command, executor)
            if process.returncode != 0:
                raise RuntimeError(f'Docker preparation failed with exit status {process.returncode}')
            if any(results.iterdir()):
                raise RuntimeError('Verifier results directory must start empty')
            process = _run(verify_command, executor)
            if process.returncode not in (0, 1):
                raise RuntimeError(f'Docker verification failed with exit status {process.returncode}')
            result = _json(_read_regular(results / 'result.json', MAX_RESULT))
            _validate_result(result, process.returncode)
            if (_json(_read_regular(results / 'metrics.json', MAX_RESULT)) != result
                    or _json(process.stdout) != result):
                raise ValueError('Verifier stdout and fresh result files disagree')
            # Publish only validated JSON, never arbitrary container-created files.
            _write_json(output / 'result.json', result)
            _write_json(output / 'metrics.json', result)
        print(json.dumps(result, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        result = {'status': 'error', 'error': type(error).__name__ + ': ' + str(error)}
        _log(result['error'])
        if output is not None:
            # A partial publication is never presented as a usable grade.
            for name in ('result.json', 'metrics.json'):
                (output / name).unlink(missing_ok=True)
            try:
                _write_json(output / 'error.json', result)
            except OSError as output_error:
                _log('Unable to retain error.json: ' + str(output_error))
        print(json.dumps(result, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
