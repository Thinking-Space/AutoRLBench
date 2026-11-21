"""Immutable dataset pins and preparation with no third-party runtime or network."""
from collections import Counter
import pytest

from tests.justrl2_review.fixtures import (
    ENGINE, ENVIRONMENT, VERIFIER, ReviewRun, Scene, inventory, load, read, write,
)


def test_bundled_data_matches_complete_pin_and_real_balanced_selection(tmp_path):
    case = Scene(tmp_path)
    pin = read(case.snapshot)
    assert pin['schema_version'] == 1
    assert pin['files'] == inventory(case.scene)
    assert len(pin['files']) == 69
    assert sum(entry['size'] for entry in pin['files'].values()) == 141589
    assert all(name.startswith('dataset/') for name in pin['files'])
    assert read(VERIFIER / 'resources/snapshot.json') == pin
    assert len(case.rows) == len({row['id'] for row in case.rows}) == 64
    assert Counter(row['difficulty'] for row in case.rows) == {'easy': 32, 'hard': 32}
    assert len((case.dataset / 'eval.jsonl').read_text().splitlines()) == 16
    assert len(case.catalog) == 64
    catalog = {row['id']: row for row in case.catalog}
    for row in case.rows:
        current = catalog[row['id']]
        assert current['sha256'] == row['sha256']
        payload = read(case.dataset / current['path'])
        assert payload['id'] == row['id']
        assert payload['metadata']['difficulty'] == row['difficulty']
        assert payload['prompt'] and 'label' in payload
    assert isinstance(load(ENGINE / 'dataset.py').verify_dataset(case.scene, case.snapshot), dict)


@pytest.mark.parametrize('damage', ['same_size_bytes', 'file_symlink', 'directory_symlink'])
def test_dataset_rejects_changed_bytes_and_identical_content_links(tmp_path, damage):
    case = Scene(tmp_path)
    payload = case.dataset / case.catalog[0]['path']
    if damage == 'same_size_bytes':
        raw = payload.read_bytes()
        payload.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif damage == 'file_symlink':
        target = tmp_path / 'outside.json'
        payload.replace(target)
        payload.symlink_to(target)
    else:
        parent = payload.parent
        target = tmp_path / 'outside-directory'
        parent.replace(target)
        parent.symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        load(ENGINE / 'dataset.py').verify_dataset(case.scene, case.snapshot)
    result = case.check()
    assert result['score'] is False and result['errors']


def test_wrong_snapshot_pin_is_rejected_even_when_dataset_is_untouched(tmp_path):
    case = Scene(tmp_path)
    pin = read(case.snapshot)
    pin['files']['dataset/manifest.json']['sha256'] = '0' * 64
    write(case.snapshot, pin)
    with pytest.raises((ValueError, OSError)):
        load(ENGINE / 'dataset.py').verify_dataset(case.scene, case.snapshot)
    result = case.check()
    assert result['score'] is False and result['errors']


def test_real_prepare_is_offline_stdlib_and_preserves_dataset_bytes(tmp_path):
    before = inventory(ENVIRONMENT / 'scene')
    run = ReviewRun(tmp_path / 'review')
    assert inventory(run.prepared / 'incident') == before
    assert inventory(ENVIRONMENT / 'scene') == before
    assert (run.workspace / 'incident').is_symlink()
    assert (run.workspace / 'incident').resolve() == (run.prepared / 'incident').resolve()
    assert inventory(run.prepared / 'incident_engine') == inventory(ENGINE)
    assert read(run.prepared / 'snapshot.json') == read(ENVIRONMENT / 'snapshot.json')
    assert {path.name for path in run.prepared.iterdir()} == {'incident', 'incident_engine', 'snapshot.json'}
