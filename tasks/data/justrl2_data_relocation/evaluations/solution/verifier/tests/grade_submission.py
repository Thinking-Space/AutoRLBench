#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Require a correct starting verdict and freshly executed data-reading checks."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def assess_report(results, workspace, resources):
    """Assess the starting implementation; later repairs do not change its verdict."""
    assessment = {'reported': None, 'expected': None, 'verdict_correct': False,
                  'report_valid': False, 'report_path': None, 'errors': []}
    try:
        metadata = json.loads((resources / 'expected_verdict.json').read_text(encoding='utf-8'))
        if (not isinstance(metadata, dict) or type(metadata.get('schema_version')) is not int
                or metadata['schema_version'] != 1
                or metadata.get('expected_verdict') not in ('healthy', 'unhealthy')):
            raise ValueError('Expected verdict metadata requires schema_version 1 and healthy/unhealthy')
        assessment['expected'] = metadata['expected_verdict']
    except (OSError, ValueError, TypeError) as error:
        assessment['errors'].append('Expected verdict: ' + str(error))
    try:
        path = workspace / 'report.md'
        assessment['report_path'] = str(path)
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Report must be a regular file, not a symlink')
        if path.stat().st_size > 65536:raise ValueError('Report exceeds 64 KiB')
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        first = next((index for index, line in enumerate(lines) if line.strip()), None)
        if first is None:
            raise ValueError('Report is empty')
        verdict = {'Verdict: healthy': 'healthy', 'Verdict: unhealthy': 'unhealthy'}.get(lines[first].strip())
        if verdict is None:
            raise ValueError('First nonblank line must be Verdict: healthy or Verdict: unhealthy')
        assessment['reported'] = verdict
        if not any(line.strip() for line in lines[first + 1:]):
            raise ValueError('Report requires a nonempty evidence body after the verdict')
        assessment['report_valid'] = True
    except (OSError, ValueError, TypeError) as error:
        assessment['errors'].append('Report: ' + str(error))
    assessment['verdict_correct'] = (assessment['reported'] is not None
                                     and assessment['reported'] == assessment['expected'])
    return assessment


def write_result(path, value):
    # Replace agent-created output links rather than following them.
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    results = Path(os.environ.get('AUTORLBENCH_RESULTS', '/results'))
    prepared = Path(os.environ.get('AUTORLBENCH_PREPARED', '/autorlbench/prepared'))
    workspace = Path(os.environ.get('AUTORLBENCH_WORKSPACE', str(Path.cwd())))
    resources = Path(__file__).resolve().parent.parent / 'resources'
    assessment = assess_report(results, workspace, resources)
    # Results and diagnostic-run output are agent-writable. Never use them as
    # proof: rerun the trusted read-only checker against the submitted source.
    try:
        process = subprocess.run([
            sys.executable, '-I', '-S', str(prepared / 'incident_engine/check_reads.py'),
            '--scene', str(prepared / 'incident'), '--loader', str(workspace / 'loader.py'),
            '--snapshot', str(resources / 'snapshot.json')],
            capture_output=True, text=True, timeout=90, cwd='/',
            env={'PATH': os.defpath, 'LANG': 'C.UTF-8'})
        proof = json.loads(process.stdout)
        if (not isinstance(proof, dict) or type(proof.get('score')) is not bool
                or process.returncode != (0 if proof['score'] else 1)):
            raise ValueError('Invalid checker result or exit status')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        proof = {'score': False, 'errors': [type(error).__name__ + ': ' + str(error)[:300]]}
    passed = (proof.get('score') is True and assessment['report_valid']
              and assessment['verdict_correct'])
    result = {'passed': passed, 'score': int(passed), 'metrics': proof, 'assessment': assessment}
    results.mkdir(parents=True, exist_ok=True)
    write_result(results / 'result.json', result)
    write_result(results / 'metrics.json', result)
    print(json.dumps(result, allow_nan=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
