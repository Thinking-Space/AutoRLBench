"""Cache-hit trust behavior of docker/autorlbench_fetch.py.

A present cache is trusted by default (no size/sha256 re-check) because
re-hashing multi-GB weights on every prepare is the dominant cost of a
warm-cache run; --recheck-cache restores the full check on a hit. A fresh
download is always verified regardless. These tests lock that contract without
touching the network: require_verified / sha256_file are spied on, and
materialize is stubbed so no bytes are copied.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "autorlbench_fetch.py"
_spec = importlib.util.spec_from_file_location("autorlbench_fetch", _SCRIPT)
assert _spec and _spec.loader
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def _hf_args(**over):
    argv = [
        "hf",
        "--repo",
        "org/model",
        "--revision",
        "a" * 40,
        "--dest",
        over.pop("dest", "/tmp/dest"),
        "--expect",
        "model.safetensors=deadbeef",
        "--expect-size",
        "model.safetensors=10",
    ]
    if over.pop("recheck", False):
        argv.append("--recheck-cache")
    return fetch.build_parser().parse_args(argv)


def _url_args(tmp, **over):
    argv = [
        "url",
        "--url",
        "https://example.org/x.bin",
        "--sha256",
        over.pop("sha", "f" * 64),
        "--dest",
        str(tmp / "dest.bin"),
    ]
    if over.pop("recheck", False):
        argv.append("--recheck-cache")
    return fetch.build_parser().parse_args(argv)


class HfCacheHitTests(unittest.TestCase):
    def _seed_cache(self, tmp: Path) -> None:
        fetch.MODELS_ROOT = tmp / "models"
        cached = fetch.hf_cache_dir("org/model", "a" * 40, "model")
        cached.mkdir(parents=True)
        (cached / "model.safetensors").write_bytes(b"0123456789")

    def test_hit_trusted_by_default_skips_verification(self):
        with mock.patch.object(fetch, "materialize") as materialize, mock.patch.object(
            fetch, "require_verified"
        ) as verify:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                self._seed_cache(Path(td))
                fetch.fetch_hf(_hf_args(dest=str(Path(td) / "dest")))
        verify.assert_not_called()
        materialize.assert_called_once()

    def test_recheck_cache_verifies_on_hit(self):
        with mock.patch.object(fetch, "materialize"), mock.patch.object(
            fetch, "require_verified"
        ) as verify:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                self._seed_cache(Path(td))
                fetch.fetch_hf(_hf_args(dest=str(Path(td) / "dest"), recheck=True))
        verify.assert_called_once()


class UrlCacheHitTests(unittest.TestCase):
    def _seed_blob(self, tmp: Path, sha: str) -> None:
        fetch.DATASETS_ROOT = tmp / "datasets"
        blob = fetch.DATASETS_ROOT / "url" / sha.lower() / "blob"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"cached-bytes")

    def test_hit_trusted_by_default_skips_hashing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            fetch, "sha256_file"
        ) as hasher:
            tmp = Path(td)
            self._seed_blob(tmp, "f" * 64)
            fetch.fetch_url(_url_args(tmp))
        hasher.assert_not_called()

    def test_recheck_cache_hashes_on_hit(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            fetch, "sha256_file", return_value="f" * 64
        ) as hasher:
            tmp = Path(td)
            self._seed_blob(tmp, "f" * 64)
            fetch.fetch_url(_url_args(tmp, recheck=True))
        hasher.assert_called_once()


class LocalBlobReuseTests(unittest.TestCase):
    """Content-addressed reuse: a byte-identical blob already in the shared cache
    satisfies a sha-pinned file without the network. This is the only working path
    for an xet-stored weight on a cluster whose egress proxy blocks the xet CDN.
    """

    def _sibling_blob(self, tmp: Path, sha: str, data: bytes) -> Path:
        fetch.MODELS_ROOT = tmp / "models"
        blob = fetch.MODELS_ROOT / "hf" / "hub" / "models--other--repo" / "blobs" / sha
        blob.parent.mkdir(parents=True)
        blob.write_bytes(data)
        return blob

    def test_find_local_blobs_matches_by_sha_and_size(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            blob = self._sibling_blob(tmp, "deadbeef", b"0123456789")
            found = fetch.find_local_blobs(
                "model", {"model.safetensors": "deadbeef"}, {"model.safetensors": 10}
            )
            self.assertEqual(found, {"model.safetensors": blob})

    def test_find_local_blobs_rejects_size_mismatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._sibling_blob(tmp, "deadbeef", b"0123456789")
            found = fetch.find_local_blobs(
                "model", {"model.safetensors": "deadbeef"}, {"model.safetensors": 11}
            )
            self.assertEqual(found, {})

    def test_seed_hf_cache_builds_pointer_to_real_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            blob = self._sibling_blob(tmp, "deadbeef", b"0123456789")
            rev = "a" * 40
            fetch.seed_hf_cache("org/model", rev, "model", {"model.safetensors": blob})
            pointer = fetch.hf_cache_dir("org/model", rev, "model") / "model.safetensors"
            # The pointer resolves (not dangling) to the right 10 bytes.
            self.assertTrue(pointer.is_symlink())
            self.assertEqual(pointer.read_bytes(), b"0123456789")
            # And it lives in this repo's own blobs/, keyed by the sha (= etag).
            own_blob = (
                fetch.hf_repo_dir("org/model", "model") / "blobs" / "deadbeef"
            )
            self.assertTrue(own_blob.is_file())


if __name__ == "__main__":
    unittest.main()
