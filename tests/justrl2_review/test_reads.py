"""Execute actual bounded queries against the real 64-row frozen dataset."""
import json
import sys

import pytest

from tests.justrl2_review.fixtures import (
    ENGINE, TASK, REFERENCE_LOADER, Scene, assert_reads_pass, execute, inventory,
)


def test_reference_reads_all_original_and_randomized_records_without_mutation(tmp_path):
    case = Scene(tmp_path)
    before = inventory(case.scene)
    assert_reads_pass(case.check())
    assert inventory(case.scene) == before


@pytest.mark.parametrize('implementation', ['faulty', 'memorized', 'wrong_record', 'constant', 'always_raise'])
def test_nonrepairs_cannot_pass_real_read_acceptance(tmp_path, implementation):
    case = Scene(tmp_path)
    first = case.catalog[0]
    sources = {
        'faulty': (TASK / 'workspace/loader.py').read_text(),
        'memorized': 'def resolve(root, record):\n    return record["path"].replace("live/hard/", "archive/hard/")\n',
        'wrong_record': '''import json
from pathlib import Path
def resolve(root, record):
    rows = json.loads((Path(root) / 'catalog.json').read_text())['rows']
    candidates = [row for row in rows if row['id'] != record['id']]
    return candidates[0]['path']
''',
        'constant': f'def resolve(root, record):\n    return {first["path"]!r}\n',
        'always_raise': 'def resolve(root, record):\n    raise FileNotFoundError("unavailable")\n',
    }
    result = case.check(sources[implementation])
    assert result['score'] is False, result
    assert result['errors']
    assert any(not entry['passed'] for entry in result['variants'])
    if implementation == 'memorized':
        variants = {entry['name']: entry for entry in result['variants']}
        assert variants['original']['readable_count'] == 64
        assert variants['relocated']['passed'] is False


@pytest.mark.parametrize('shortcut', ['existing_path', 'missing_file', 'duplicate_catalog', 'zero_hash_special_case'])
def test_positive_reads_do_not_replace_unavailable_record_probes(tmp_path, shortcut):
    case = Scene(tmp_path)
    source = REFERENCE_LOADER.read_text()
    if shortcut == 'existing_path':
        source = source.replace('    root = Path(root)\n',
            '    root = Path(root)\n    if (root / record["path"]).is_file():\n        return record["path"]\n')
    elif shortcut == 'missing_file':
        source = source.replace("    if not (root / path).is_file():\n        raise FileNotFoundError('Current backing file is unavailable')\n", '')
    elif shortcut == 'duplicate_catalog':
        source = source.replace('if len(candidates) != 1:', 'if len(candidates) == 0:')
    else:
        source = source.replace("and row['sha256'] == record['sha256']", '')
        source = source.replace('    root = Path(root)\n',
            '    root = Path(root)\n    if record["sha256"] == "' + '0' * 64 + '":\n        raise FileNotFoundError("missing")\n')
    assert source != REFERENCE_LOADER.read_text(), 'Reference shape changed; update the deliberate mutation'
    result = case.check(source)
    assert result['score'] is False, result
    assert result['errors']
    assert any(not probe['passed'] for probe in result['probes'])
    if shortcut != 'existing_path':
        assert all(entry['passed'] for entry in result['variants'])


@pytest.mark.parametrize('repair', [False, True])
def test_trusted_cli_returns_only_json_and_matching_exit_status(tmp_path, repair):
    case = Scene(tmp_path)
    if not repair:
        case.loader.write_bytes((TASK / 'workspace/loader.py').read_bytes())
    process = execute([sys.executable, '-I', '-S', '-B', ENGINE / 'check_reads.py',
                       '--scene', case.scene, '--loader', case.loader,
                       '--snapshot', case.snapshot], cwd=tmp_path)
    assert process.returncode == (0 if repair else 1), process.stderr
    result = json.loads(process.stdout)
    assert result['score'] is repair
    if repair:
        assert_reads_pass(result)
