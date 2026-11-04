"""Prepare original source snapshots plus pinned regression tests, never repaired production code."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, metadata):
    assert path.stat().st_size == metadata["bytes"], path
    assert digest(path) == metadata["sha256"], path


def build(root):
    corpus = root / "data/corpus"
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    for record in manifest["cases"]:
        target = corpus / "cases" / record["case_id"] / "base"
        marker = target / ".complete"
        tests = record["test_file"].split(";")
        if (marker.exists() and marker.read_text() == manifest_hash
                and digest(target / record["source_file"]) == record["source_sha256"]
                and all(digest(target / test) == manifest["test_files"][record["case_id"] + '/' + test]["sha256"] for test in tests)):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as temp:
            name = record["repo"].replace("/", "_") + "-" + record["buggy_commit_id"] + ".tar.gz"
            archive = root / ".corpus_archives" / name
            verify(archive, manifest["archives"][name])
            with tarfile.open(archive) as tar:
                tar.extractall(temp, filter="data")
            staged = next(Path(temp).iterdir())
            for test in tests:
                key = record["case_id"] + '/' + test
                source = root / ".corpus_tests" / key
                verify(source, manifest["test_files"][key])
                (staged / test).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, staged / test)
            assert digest(staged / record["source_file"]) == record["source_sha256"]
            (staged / ".complete").write_text(manifest_hash)
            if target.exists():
                shutil.rmtree(target)
            staged.rename(target)


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)
