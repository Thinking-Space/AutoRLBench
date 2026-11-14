import json
from pathlib import Path
import shutil
import subprocess

import grade_debugging
import replay_probe
from test_runtime_verifier import exported
from verify_runtime import CASES


def test_patch_without_debugging_evidence_fails(tmp_path):
    manifest = tmp_path / 'submission.json'
    manifest.write_text('{}')
    assert not grade_debugging.validate(manifest, tmp_path, tmp_path)['passed']


def test_real_experiment_gate_and_missing_controls(exported, tmp_path):
    original = exported / 'workspace'
    repaired = tmp_path / 'repaired'
    shutil.copytree(original, repaired)
    subprocess.run(['git', 'apply', str(exported / 'maintainer/repair.patch')], cwd=repaired, check=True)
    before = ['cold_short', 'warm_short', 'long', 'unequal_dp3']
    after = before + ['cp1']
    payload = {'before': [], 'after': [], 'report': 'report.md'}
    for phase, names, source in [('before', before, original), ('after', after, repaired)]:
        for name in names:
            directory = tmp_path / f'{phase}-{name}'
            replay_probe.run_probe(source, directory, **CASES[name], trace=True)
            payload[phase].append(str(directory))
    (tmp_path / 'report.md').write_text('Runtime evidence is supplied; semantic diagnosis still requires reviewer judgment.\n')
    manifest = tmp_path / 'submission.json'
    manifest.write_text(json.dumps(payload))
    result = grade_debugging.validate(manifest, original, repaired)
    assert result['passed'], result
    summary = tmp_path / 'before-warm_short/summary.json'
    saved_summary = summary.read_text()
    changed = json.loads(saved_summary)
    changed['config']['world_size'] = 1
    summary.write_text(json.dumps(changed))
    result = grade_debugging.validate(manifest, original, repaired)
    assert not result['passed'] and 'World size' in result['errors'][0]
    summary.write_text(saved_summary)
    row = tmp_path / 'before-warm_short/rank0.json'
    saved_row = row.read_text()
    changed = json.loads(saved_row)
    changed['steps'][0]['advantages'][0][0] += 1
    row.write_text(json.dumps(changed))
    result = grade_debugging.validate(manifest, original, repaired)
    assert not result['passed'] and 'completed-prefix' in result['errors'][0]
    row.write_text(saved_row)
    payload['before'] = payload['before'][:2]
    manifest.write_text(json.dumps(payload))
    result = grade_debugging.validate(manifest, original, repaired)
    assert not result['passed'] and 'controls' in result['errors'][0]
