"""Six preparation/materialization/public-pin contracts; no network or models."""
import json

import pytest

from tests.justrl2_review.fixtures import FAULTY, Fixture, inventory, load, write
from tests.justrl2_review.public.fixtures import PublicFixture


def materializer(monkeypatch):
    root = FAULTY / 'environment/incident_engine'
    monkeypatch.syspath_prepend(str(root))
    return load(root / 'materialize.py')


def test_scene_materialization_preserves_all_bytes_and_stale_manifest(tmp_path, monkeypatch):
    case = Fixture(tmp_path / 'source')
    before = inventory(case.scene)
    workspace, prepared = tmp_path / 'workspace', tmp_path / 'prepared'
    workspace.mkdir(); prepared.mkdir()
    pin = tmp_path / 'snapshot.json'
    write(pin, {'scene_sha256': before['SCENE.json']['sha256']})
    installed = materializer(monkeypatch).install(case.scene, workspace, prepared, pin)
    assert inventory(installed) == before == inventory(case.scene)
    assert (workspace / 'incident').resolve() == installed.resolve()
    assert not (installed / 'dataset/live/hard').exists()
    assert load(case.bundle / 'scene.py').verify_scene(installed)['sample_cursor'] == 32


def test_wrong_snapshot_pin_fails_before_installing_any_scene(tmp_path, monkeypatch):
    case = Fixture(tmp_path / 'source')
    workspace, prepared = tmp_path / 'workspace', tmp_path / 'prepared'
    workspace.mkdir(); prepared.mkdir()
    pin = tmp_path / 'snapshot.json'
    write(pin, {'scene_sha256': '0' * 64})
    with pytest.raises(ValueError, match='pin'):
        materializer(monkeypatch).install(case.scene, workspace, prepared, pin)
    assert not (prepared / 'incident').exists()


def test_scene_archive_roundtrip_retains_complete_inventory(tmp_path, monkeypatch):
    case = Fixture(tmp_path / 'source')
    scene = load(case.bundle / 'scene.py')
    archive = tmp_path / 'snapshot.tar'
    packed = scene.pack_scene(case.scene, archive)
    pin = tmp_path / 'snapshot.json'
    write(pin, dict(scene_sha256=packed['scene_sha256'], archive_sha256=packed['sha256'], archive_size=packed['size']))
    workspace, prepared = tmp_path / 'workspace', tmp_path / 'prepared'
    workspace.mkdir(); prepared.mkdir()
    installed = materializer(monkeypatch).install(archive, workspace, prepared, pin)
    assert inventory(installed) == inventory(case.scene)


def test_public_preparation_matches_selected_bytes_and_provenance(tmp_path):
    fixture = PublicFixture(tmp_path / 'public')
    receipt = fixture.prepare()
    for name, expected in fixture.expected.items():
        assert (fixture.out / name).read_bytes() == expected
    assert receipt['runtime_sha256'] == fixture.selection['expected_runtime_sha256']
    case = json.loads((fixture.out / 'CASE.json').read_text())
    assert case['source']['repo'] == fixture.lock['repo']
    assert [row['id'] for row in case['rows']] == [row['id'] for row in fixture.selection['rows']]


def test_public_source_hash_change_is_rejected_before_output(tmp_path):
    fixture = PublicFixture(tmp_path / 'public')
    source = fixture.sources / 'Math_part-1-of-4.jsonl'
    source.write_bytes(source.read_bytes().replace(b'Synthetic', b'synthetic'))
    with pytest.raises(ValueError, match='SHA256'):
        fixture.prepare()
    assert not fixture.out.exists()


def test_prepared_model_inventory_rejects_same_size_byte_change(tmp_path):
    prepare = load(FAULTY / 'environment/asset_tools/prepare_assets.py')
    model = tmp_path / 'model'
    model.mkdir()
    path = model / 'model.safetensors'
    path.write_bytes(b'SYNTHETIC unloadable bytes')
    pins = inventory(model)
    assert prepare.verify_inventory(model, pins) == pins
    path.write_bytes(b'X' * pins[path.name]['size'])
    with pytest.raises(ValueError, match='hash|SHA256'):
        prepare.verify_inventory(model, pins)
