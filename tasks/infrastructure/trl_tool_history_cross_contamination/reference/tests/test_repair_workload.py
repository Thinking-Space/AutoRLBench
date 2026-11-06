import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workspace"))

"""Integration checks against the frozen real upstream corpus (no model required)."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from reproductions.trl_history.environment import RepairEnvironment


@pytest.fixture
def env(tmp_path):
    corpus = os.environ.get("PC_REPAIR_CORPUS")
    python = os.environ.get("PC_CASE_PYTHON")
    if not corpus or not python:
        pytest.skip("Set PC_REPAIR_CORPUS and PC_CASE_PYTHON for the real-corpus integration tests")
    environment = RepairEnvironment(Path(corpus), python, tmp_path / "traces")
    yield environment
    if environment.work:
        shutil.rmtree(environment.work)


def apply_reference(environment):
    subprocess.run(["git", "apply", str(Path(__file__).resolve().parents[1] / "corpus_patches" / (environment.case.name + ".patch"))], cwd=environment.work,
                   env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(environment.work.parent)), check=True)


def test_all_original_bugs_fail_and_reference_patches_pass(env):
    for case_id in sorted(env.records):
        env.reset(case_id)
        defective = env._test()
        assert not defective["passed"], (case_id, defective)
        assert defective["counts"]["failures"] > 0, (case_id, defective)
        apply_reference(env)
        repaired = env._test()
        assert repaired["passed"], (case_id, repaired)


def test_four_identical_resets_have_independent_workspaces(env, tmp_path):
    peers = [RepairEnvironment(env.corpus, env.python, tmp_path / f"peer-{i}") for i in range(4)]
    try:
        observations = [peer.reset("thefuck-32") for peer in peers]
        assert len(set(observations)) == 1
        assert len({peer.work for peer in peers}) == 4
        initial = peers[0].target.read_bytes()
        apply_reference(peers[0])
        assert peers[0].target.read_bytes() != initial
        assert all(peer.target.read_bytes() == initial for peer in peers[1:])
        assert peers[0].reset("thefuck-32") == observations[0]
        assert peers[0].target.read_bytes() == initial
    finally:
        for peer in peers:
            shutil.rmtree(peer.work)


def test_grader_rebuilds_protected_tests(env):
    env.reset("thefuck-32")
    (env.work / "pytest.ini").write_text("[pytest]\naddopts = --collect-only\n")
    test = env.work / "tests/rules/test_ls_lah.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_fake(): pass\n")
    assert env.get_reward() == 0
    apply_reference(env)
    assert env.get_reward() == 1


def test_budget_and_rejected_edit(env):
    env.reset("thefuck-32")
    original = env.target.read_bytes()
    assert "rejected" in env.replace_source("missing text", "replacement")
    assert env.target.read_bytes() == original
    for _ in range(5):
        env.inspect_source()
    with pytest.raises(ValueError, match="budget"):
        env.run_tests()
    apply_reference(env)
    assert env.get_reward() == 0


def test_source_groups_never_cross_splits(env):
    groups = {}
    manifest = json.loads((env.corpus / "manifest.json").read_text())
    assert {key: len(value) for key, value in manifest["splits"].items()} == {"train": 16, "dev": 4, "test": 8}
    for record in manifest["cases"]:
        key = record["project"], record["source_file"]
        assert groups.setdefault(key, record["split"]) == record["split"]
