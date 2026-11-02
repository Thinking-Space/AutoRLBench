#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Require a starting-state review and fresh checkpoint-continuation evidence."""
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
from pathlib import Path


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
        archived = results / 'report.md'
        path = archived if archived.exists() or archived.is_symlink() else workspace / 'report.md'
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
    results=Path(os.environ.get('AUTORLBENCH_RESULTS','/results'))
    prepared=Path(os.environ.get('AUTORLBENCH_PREPARED','/autorlbench/prepared'))
    workspace=Path(os.environ.get('AUTORLBENCH_WORKSPACE',str(Path.cwd())))
    resources=Path(__file__).resolve().parent.parent/'resources'
    assessment=assess_report(results,workspace,resources)
    try:
        path=resources/'incident_audit/verify_resume.py'
        spec=importlib.util.spec_from_file_location('_hidden_incident_resume',path)
        engine=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        pin=json.loads((resources/'snapshot.json').read_text())
        scene=prepared/'incident'
        if hashlib.sha256((scene/'SCENE.json').read_bytes()).hexdigest()!=pin['scene_sha256']:
            raise ValueError('Scene pin mismatch')
        proof=engine.verify_resume(scene,results/'experiment_logs/run-home',fresh=True)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        proof={'schema_version':3,'score':False,'errors':[str(error)]}
    passed=proof.get('score') is True and assessment['report_valid'] and assessment['verdict_correct']
    result={'passed':passed,'score':int(passed),'metrics':proof,'assessment':assessment}
    results.mkdir(parents=True,exist_ok=True)
    write_result(results/'result.json',result)
    write_result(results/'metrics.json',result)
    print(json.dumps(result,allow_nan=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
