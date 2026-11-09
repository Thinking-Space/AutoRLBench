"""Seven report/continuation cases for the review grading pipeline."""
import pytest

from tests.justrl2_review.fixtures import Fixture, ReviewRun


@pytest.mark.parametrize('expected,reported,repair,accepted', [
    ('healthy', 'healthy', False, True), ('unhealthy', 'unhealthy', True, True),
    ('healthy', 'unhealthy', False, False), ('unhealthy', 'healthy', True, False)])
def test_starting_verdict_and_continuation_joint_score(tmp_path, expected, reported, repair, accepted):
    case = Fixture(tmp_path / 'case')
    run = ReviewRun(tmp_path / 'review', expected, case)
    run.submit(f'\nVerdict: {reported}\n\nSYNTHETIC evidence about the initial resolver.\n', repair=repair)
    result, code = run.run()
    assert result['score'] == int(accepted) and result['passed'] is accepted
    assert result['metrics']['score'] is True and result['metrics']['admitted_count'] == 64
    assert (run.loader.read_bytes() != run.initial_loader) is repair


def test_missing_report_does_not_overwrite_a_successful_original_proof(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    run.submit()
    result, _ = run.run()
    assert result['score'] == 0 and result['metrics']['score'] is True
    assert result['assessment']['reported'] is None and result['assessment']['errors']


def test_archived_report_without_evidence_never_falls_back_to_workspace(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy', Fixture(tmp_path / 'case'))
    run.submit('Verdict: healthy\n\n')
    (run.workspace / 'report.md').write_text('Verdict: healthy\nSYNTHETIC newer evidence.\n')
    result, _ = run.run()
    assert result['score'] == 0 and result['assessment']['report_valid'] is False
    assert result['assessment']['report_path'] == str(run.results / 'report.md')


def test_healthy_report_never_bypasses_fresh_checkpoint_requirement(tmp_path):
    case = Fixture(tmp_path / 'case')
    run = ReviewRun(tmp_path / 'review', 'healthy', case)
    run.submit('Verdict: healthy\nSYNTHETIC unchanged implementation.\n', archived=False)
    (case.home / 'run/actor/iter_0000007/__0_0.distcp').unlink()
    result, _ = run.run()
    assert result['score'] == 0 and result['metrics']['score'] is False
    assert result['assessment']['verdict_correct'] is True
