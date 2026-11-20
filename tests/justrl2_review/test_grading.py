"""Starting verdict and fresh data reads jointly decide the solution score."""
import pytest

from tests.justrl2_review.fixtures import ReviewRun, assert_reads_pass, write


@pytest.mark.parametrize('expected,reported,repair,accepted', [
    ('healthy', 'healthy', False, True),
    ('unhealthy', 'unhealthy', True, True),
    ('healthy', 'unhealthy', False, False),
    ('unhealthy', 'healthy', True, False),
])
def test_starting_verdict_and_current_reads_joint_score(tmp_path, expected, reported, repair, accepted):
    run = ReviewRun(tmp_path / 'review', expected)
    run.submit(reported, repair=repair)
    result = run.run()
    assert result['passed'] is accepted
    assert_reads_pass(result['metrics'])
    assert result['assessment']['verdict_correct'] is accepted
    assert (run.loader.read_bytes() != run.initial_loader) is repair


def test_correct_unhealthy_verdict_cannot_excuse_unrepaired_loader(tmp_path):
    run = ReviewRun(tmp_path / 'review')
    run.submit('unhealthy')
    result = run.run()
    assert result['score'] == 0 and result['metrics']['score'] is False
    assert result['assessment']['verdict_correct'] is True


def test_missing_workspace_report_is_not_replaced_by_archived_report(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy')
    (run.results / 'report.md').write_text('Verdict: healthy\nArchived evidence.\n')
    result = run.run()
    assert result['score'] == 0
    assert_reads_pass(result['metrics'])
    assert result['assessment']['reported'] is None
    assert result['assessment']['report_valid'] is False
    assert result['assessment']['errors']


def test_current_report_wins_over_conflicting_results_and_needs_no_magic_evidence(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy')
    run.report.write_text('Verdict: healthy\n\nx\n')
    (run.results / 'report.md').write_text('Verdict: unhealthy\nContradictory old report.\n')
    result = run.run()
    assert result['score'] == 1
    assert result['assessment']['report_path'] == str(run.report)
    assert result['assessment']['reported'] == 'healthy'


def test_forged_success_is_recomputed_and_post_success_changes_are_rechecked(tmp_path):
    run = ReviewRun(tmp_path / 'review')
    run.submit('unhealthy', repair=True)
    real_success = run.run()
    assert real_success['score'] == 1
    # Reuse a genuine old success to ensure a plausible forged schema cannot help.
    for name in ('result.json', 'metrics.json', 'check_reads.json', 'experiment_logs/check_reads.json'):
        write(run.results / name, real_success)
    run.loader.write_text('def resolve(root, record):\n    raise FileNotFoundError("all missing")\n')
    failed = run.run()
    assert failed['score'] == 0 and failed['metrics']['score'] is False
    assert failed['metrics']['errors']


@pytest.mark.parametrize('artifact', ['loader.py', 'report.md'])
def test_submission_links_are_rejected_without_reading_link_target(tmp_path, artifact):
    run = ReviewRun(tmp_path / 'review', 'healthy')
    run.submit('healthy')
    path = run.workspace / artifact
    target = tmp_path / ('outside-' + artifact)
    path.replace(target)
    path.symlink_to(target)
    before = target.read_bytes()
    result = run.run()
    assert result['score'] == 0
    assert target.read_bytes() == before
    if artifact == 'report.md':
        assert result['assessment']['report_valid'] is False
        assert result['assessment']['reported'] is None
    else:
        assert result['metrics']['score'] is False


def test_output_links_are_atomically_replaced_without_overwriting_targets(tmp_path):
    run = ReviewRun(tmp_path / 'review', 'healthy')
    run.submit('healthy')
    targets = {}
    for name in ('result.json', 'metrics.json'):
        targets[name] = tmp_path / ('outside-' + name)
        targets[name].write_text('unchanged')
        (run.results / name).symlink_to(targets[name])
    result = run.run()
    assert result['score'] == 1
    for name, target in targets.items():
        assert target.read_text() == 'unchanged'
        assert not (run.results / name).is_symlink()
