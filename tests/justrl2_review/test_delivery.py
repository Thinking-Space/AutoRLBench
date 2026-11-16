"""Six delivery/control and unavailable-record regressions; no model execution."""
import shutil

import pytest

from tests.justrl2_review.fixtures import FAULTY, Fixture, REFERENCE_LOADER, ReviewRun, load


def test_generated_control_is_standalone_and_accepts_unchanged_code(tmp_path):
    from autorlbench.tasks.loader import load_task_package
    generator = load(FAULTY / 'provenance/make_control.py')
    control = generator.make_control(tmp_path / 'control')
    assert load_task_package(control).metadata['outcome']['failure_type'] == 'healthy_control'
    for tree in ('environment', 'workspace/logs'):
        for source in (FAULTY / tree).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                assert (control / source.relative_to(FAULTY)).read_bytes() == source.read_bytes()
    for level in ('issue', 'solution'):
        relative = f'evaluations/{level}/instruction.md'
        assert (control / relative).read_bytes() == (FAULTY / relative).read_bytes()
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    shutil.copyfile(control / 'workspace/loader.py', run.loader)
    shutil.copyfile(control / 'evaluations/solution/verifier/tests/grade_submission.py', run.wrapper)
    shutil.copyfile(control / 'evaluations/solution/verifier/resources/expected_verdict.json',
                    run.wrapper.parent.parent / 'resources/expected_verdict.json')
    initial = run.loader.read_bytes()
    run.submit('Verdict: healthy\nSYNTHETIC current records verified without a code change.\n')
    result, code = run.run()
    assert code == 0 and result['score'] == 1 and run.loader.read_bytes() == initial
    with pytest.raises(FileExistsError):
        generator.make_control(control)


@pytest.mark.parametrize('unavailable', ['unknown_existing_path', 'wrong_hash_existing_path'])
def test_reference_rejects_unavailable_records_on_both_storage_layouts(tmp_path, unavailable):
    case = Fixture(tmp_path / 'case')
    root = case.scene / 'dataset'
    worker = load(FAULTY / 'environment/incident_engine/resolve_worker.py')
    record = dict(next(row for row in case.rows if row['difficulty'] == 'easy'))
    if unavailable == 'unknown_existing_path':
        record['id'] = 'unknown-id'
    else:
        record['sha256'] = '0' * 64
    with pytest.raises((FileNotFoundError, ValueError)):
        worker.Query(root).run(REFERENCE_LOADER.read_text(), record)


@pytest.mark.parametrize('attack', ['missing_worker', 'existing_path_shortcut', 'missing_current_file'])
def test_contract_probes_fail_closed_despite_synthetic_success_evidence(tmp_path, attack):
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    if attack == 'missing_worker':
        (run.prepared / 'incident_engine/resolve_worker.py').unlink()
    elif attack == 'existing_path_shortcut':
        source = REFERENCE_LOADER.read_text()
        source = source.replace('    root = Path(root)\n',
            "    root = Path(root)\n    if (root / record['path']).is_file():\n        return record['path']\n")
        run.loader.write_text(source)
    else:
        source = REFERENCE_LOADER.read_text().replace(
            "    if not (root / path).is_file():\n        raise FileNotFoundError('Current backing file is unavailable')\n", '')
        run.loader.write_text(source)
    run.submit('Verdict: healthy\nSYNTHETIC positive continuation is insufficient.\n')
    result, code = run.run()
    assert result['metrics']['score'] is True
    assert code == 1 and result['score'] == 0
    assert result['resolver_contract']['contract_valid'] is False
