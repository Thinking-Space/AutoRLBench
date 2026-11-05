"""Caching and aliasing behavior of the per-task fetch_hf.py.

The central docker/autorlbench_fetch.py is gone; each task's environment/ now
ships its own standalone copy next to prepare.sh (the image contains the task's
stack, not the harness, so the script must not import autorlbench). The copy
tested here is the ported task's; the other shipped copies differ only in
comments. Watched, without touching the network:

- a warm cache dir is trusted as a complete download (the .partial staging
  rename guarantees "exists" means "complete"), so snapshot_download is not
  called again;
- an interrupted download leaves a .partial dir, which must NOT count as a hit;
- --cache-as lands a leaky repo name under a neutral alias, and the revealing
  slug never appears on disk;
- materialize() turns the cache into real bytes at --dest, replacing whatever
  was there.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tasks"
    / "algorithm"
    / "verl_grpo_advantage_broadcast"
    / "environment"
    / "fetch_hf.py"
)
_spec = importlib.util.spec_from_file_location("task_fetch_hf", _SCRIPT)
assert _spec and _spec.loader
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def _args(tmp: Path, **over) -> types.SimpleNamespace:
    argv = [
        "--repo",
        over.pop("repo", "org/model"),
        "--dest",
        str(over.pop("dest", tmp / "dest")),
    ]
    if "type" in over:
        argv += ["--type", over.pop("type")]
    if "cache_as" in over:
        argv += ["--cache-as", over.pop("cache_as")]
    return fetch.build_parser().parse_args(argv)


def _fake_hub(download_to: Path):
    """A huggingface_hub whose snapshot_download populates local_dir."""
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"0123456789")
        return str(target)

    hub.snapshot_download = snapshot_download
    return hub


class WarmCacheTests(unittest.TestCase):
    def test_hit_is_reused_without_a_second_download(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fetch.MODELS_ROOT = tmp / "models"
            cache = fetch.repo_cache_dir("org/model", "model")
            cache.mkdir(parents=True)
            (cache / "model.safetensors").write_bytes(b"warm")

            calls = []

            def no_download(**kw):
                calls.append(kw)
                raise AssertionError("warm cache must not re-download")

            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = no_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
                fetch.fetch_hf(_args(tmp))
            self.assertEqual(calls, [])
            self.assertEqual(
                (tmp / "dest" / "model.safetensors").read_bytes(), b"warm"
            )

    def test_an_interrupted_partial_dir_is_not_a_hit(self):
        # Only a completed download is renamed into place; .partial means the
        # next run has to try again, not serve half a repo.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fetch.MODELS_ROOT = tmp / "models"
            partial = fetch.repo_cache_dir("org/model", "model").with_name(
                fetch.repo_cache_dir("org/model", "model").name + ".partial"
            )
            partial.mkdir(parents=True)
            (partial / "model.safetensors").write_bytes(b"half a file")

            with mock.patch.dict(
                sys.modules, {"huggingface_hub": _fake_hub(tmp)}
            ):
                fetch.fetch_hf(_args(tmp))
            # The stale .partial was cleared, the real cache completed, and the
            # destination carries the downloaded bytes rather than the partial ones.
            self.assertFalse(partial.exists())
            self.assertEqual(
                (tmp / "dest" / "model.safetensors").read_bytes(), b"0123456789"
            )


class CacheAsAliasTests(unittest.TestCase):
    def test_a_leaky_repo_name_never_lands_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fetch.MODELS_ROOT = tmp / "models"
            with mock.patch.dict(sys.modules, {"huggingface_hub": _fake_hub(tmp)}):
                fetch.fetch_hf(
                    _args(tmp, repo="org/answer-repo", cache_as="org/neutral-asset")
                )
            self.assertFalse(
                (fetch.MODELS_ROOT / "org__answer-repo").exists(),
                "the revealing slug must never be written to the shared cache",
            )
            alias = fetch.repo_cache_dir(
                "org/answer-repo", "model", cache_as="org/neutral-asset"
            )
            self.assertTrue((alias / "model.safetensors").is_file())

    def test_a_warm_alias_is_a_trusted_hit(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fetch.MODELS_ROOT = tmp / "models"
            with mock.patch.dict(sys.modules, {"huggingface_hub": _fake_hub(tmp)}):
                fetch.fetch_hf(
                    _args(tmp, repo="org/answer-repo", cache_as="org/neutral-asset")
                )

            def no_download(**kw):
                raise AssertionError("warm alias must not re-download")

            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = no_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
                fetch.fetch_hf(
                    _args(tmp, repo="org/answer-repo", cache_as="org/neutral-asset")
                )


class MaterializeTests(unittest.TestCase):
    def test_dest_is_real_bytes_and_replaces_what_was_there(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source = tmp / "cache" / "model"
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"real")
            dest = tmp / "workspace" / "model"
            dest.mkdir(parents=True)
            (dest / "stale-run.log").write_text("an earlier attempt", encoding="utf-8")

            fetch.materialize(source, dest)
            self.assertEqual((dest / "model.safetensors").read_bytes(), b"real")
            self.assertFalse((dest / "stale-run.log").exists())

    def test_a_dataset_routes_to_the_datasets_root(self):
        # --type picks which of the two cache roots holds the repo; routing a
        # dataset into models/ would make the next warm run miss.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fetch.MODELS_ROOT = tmp / "models"
            fetch.DATASETS_ROOT = tmp / "datasets"
            self.assertEqual(
                fetch.repo_cache_dir("org/ds", "dataset"),
                fetch.DATASETS_ROOT / "org__ds",
            )
            self.assertEqual(
                fetch.repo_cache_dir("org/model", "model"),
                fetch.MODELS_ROOT / "org__model",
            )


if __name__ == "__main__":
    unittest.main()
