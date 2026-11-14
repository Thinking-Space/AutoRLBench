#!/usr/bin/env python3
"""Fetch model weights and datasets from inside the prepare stage.

Lives in each task's `environment/` next to `prepare.sh`, which runs it directly
(`python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" …`). It runs inside the
evaluation image the task declared (docker/images/<name>.Dockerfile) and is
deliberately standalone -- no harness import -- because that image contains the
task's stack, not the harness.

What it does, kept intentionally small:

* **Cache first, by directory.** A repo is downloaded once into a per-repo dir
  under `$AUTORLBENCH_MODELS` (models) or `$AUTORLBENCH_DATASETS` (datasets). If that
  dir already exists it is reused as-is -- so a second task wanting the same repo
  does not re-download, and a rebuilt image does not either. The dir is named
  after the repo id (owner/name -> owner__name) so a warm cache is easy to read;
  a `--revision` download lands under owner__name@<rev12> so a pinned commit can
  never be served from another revision's cached tree. A repo whose name would
  give away the task's answer passes `--cache-as` to land under a neutral alias
  instead: the shared cache is mounted read-only into the agent sandbox on some
  backends, and that name must not show up there.
* **Whole repo, or an allow-list.** The full repo is fetched unless `--allow`
  globs are given; the task's own code reads whatever subset it needs and
  verifies the pinned sha256s itself (check_assets.py / prepare_models.py --
  this fetcher does not checksum).
* **Hardlink into place.** The cached tree is hardlinked to `--dest` (a copy
  across filesystems), so `--dest` is real bytes the agent/verifier can read.

xet (HF's content-addressed transfer backend) is disabled before huggingface_hub
is imported: cluster nodes reach HF only through the whitelist egress proxy /
mirror, and the xet CDN (cas-*.xethub.hf.co) is not on that path, so a
xet-enabled download stalls or fails even when the mirror is reachable.

Usage, as prepare.sh writes it:

    python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \\
        --repo Qwen/Qwen2.5-0.5B-Instruct \\
        --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \\
        --dest "$AUTORLBENCH_WORKSPACE/Qwen2.5-0.5B-Instruct"

    python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \\
        --type dataset --repo some/dataset-repo \\
        --allow "route/**" --allow "images/**" \\
        --dest "$AUTORLBENCH_WORKSPACE/data"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

# The two cache roots are the repo root's models/ and datasets/ dirs on the
# host. Defaults must match the harness's MODELS_DIR / DATASETS_DIR.
MODELS_ROOT = Path(os.environ.get("AUTORLBENCH_MODELS", "/autorlbench/models"))
DATASETS_ROOT = Path(os.environ.get("AUTORLBENCH_DATASETS", "/autorlbench/datasets"))

# Disable xet (HF's content-addressed transfer backend) before huggingface_hub
# is ever imported. Cluster nodes reach HF only through the whitelist egress
# proxy / mirror; the xet CDN (cas-*.xethub.hf.co) is not on that path, so a
# xet-enabled snapshot_download stalls or fails even when the mirror itself is
# reachable. huggingface_hub freezes this flag into a module constant at import
# time (same reason HF_HOME can't be set post-import, see fetch_hf below), so it
# MUST be set here -- at module load, before the lazy import inside fetch_hf.
# setdefault so an explicit HF_HUB_DISABLE_XET=0 in the environment still wins.
#
# NOTE: this flag alone is NOT reliable -- in huggingface_hub 0.34.x the download
# path keys off `find_spec("hf_xet")` rather than this flag (huggingface_hub#3266),
# so a xet-backed repo still hits the blocked CDN despite the export.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# The authoritative guard: make `hf_xet` unimportable in THIS process, so
# huggingface_hub's `find_spec("hf_xet")` gate sees xet as unavailable and falls
# back to plain HTTPS from the mirror. Parking a None sentinel in sys.modules
# does exactly that -- CPython's importlib.util.find_spec returns None when
# `sys.modules[name] is None` (and a bare `import hf_xet` then raises
# ImportError). It does not matter that hf_xet is still installed on the image
# (on some images it ships via conda/system packaging and can't be uninstalled):
# the sentinel makes it unavailable to the import machinery regardless, so the
# download can't route through the blocked cas-*.xethub.hf.co CDN (the SSLError
# seen in prepare). Must run before the lazy huggingface_hub import inside
# fetch_hf. Gated on the same flag so an explicit HF_HUB_DISABLE_XET=0 opt-in
# still gets real xet.
def _xet_disabled() -> bool:
    return os.environ.get("HF_HUB_DISABLE_XET", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


if _xet_disabled():
    sys.modules.setdefault("hf_xet", None)


def _force_disable_xet() -> None:
    """Overwrite every `is_xet_available` binding in huggingface_hub to False.

    Call AFTER huggingface_hub is imported. This is the version-independent
    backstop to the module-load sentinel: whatever mechanism this huggingface_hub
    uses to detect xet (find_spec, `importlib.metadata.version`, a cached module
    constant), the download path ultimately gates on `is_xet_available()`.
    Submodules bind that name by value at import time (e.g. file_download does
    `from .utils import is_xet_available`), so patching one module is not enough
    -- we walk every imported huggingface_hub.* module and replace the attribute
    wherever it exists.
    """
    if not _xet_disabled():
        return
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("huggingface_hub"):
            continue
        if hasattr(module, "is_xet_available"):
            try:
                module.is_xet_available = lambda *a, **k: False
            except (AttributeError, TypeError):
                pass


def root_for(repo_type: str) -> Path:
    """Pick the cache root for an asset kind.

    Two roots rather than one with subdirs, because on the host these are two
    side-by-side, visible directories: forgetting to mount one shows up
    immediately as "nothing was downloaded" rather than quietly writing into
    the container's writable layer.
    """
    return DATASETS_ROOT if repo_type == "dataset" else MODELS_ROOT


class FetchError(RuntimeError):
    """The asset is missing or could not be downloaded."""


# --------------------------------------------------------------------------
# materializing cache -> destination
# --------------------------------------------------------------------------


def materialize(source: Path, dest: Path) -> None:
    """Put the cached tree at `dest`, without paying for a second copy if we can.

    Hard links, falling back to a copy across filesystems. The cache roots and
    `--dest` are normally two directories on one mount, so the link path is the
    usual one and a 1 GB checkpoint costs inodes rather than gigabytes.

    Not symlinks: the link target is a container path (/autorlbench/models/...),
    and on the harbor path `--dest` is uploaded into Harbor's container as a
    plain copy of the workspace, where a symlink would dangle. `--dest` has to be
    real bytes the agent and verifier stages can read on their own.

    Any symlink inside `source` is resolved to its real target before linking
    (`os.link` does not follow symlinks on every filesystem, notably the
    Cybertron `/user` mount, and a hardlink to a symlink leaves a dangling entry
    at `dest`). Resolving to the real file keeps `dest` real bytes everywhere.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    def _link_real(src: str, dst: str) -> None:
        os.link(os.path.realpath(src), dst)

    try:
        shutil.copytree(source, dest, copy_function=_link_real, symlinks=False)
    except OSError:
        # Cross-device, or a filesystem without hard links. copytree's default
        # copy_function open()s the source, which always follows symlinks, so
        # the real bytes are copied and this path never dangles.
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, symlinks=False)


# --------------------------------------------------------------------------
# huggingface
# --------------------------------------------------------------------------


def repo_cache_dir(
    repo: str, repo_type: str, cache_as: str | None = None, revision: str | None = None
) -> Path:
    """Where this repo's download lives, keyed by the repo id (or a neutral alias).

    The dir is named after the repo id (owner/name -> owner__name) so a warm cache
    is readable. The same repo always maps to the same dir -- that is the cache
    key. Pass ``cache_as`` for a repo whose name would reveal the task's answer:
    the cache is mounted read-only into the agent sandbox on some backends, so
    that repo caches under the neutral alias while still being reused across runs.
    A pinned ``revision`` is appended (@<first 12 chars) so a pinned-commit
    download is never served from -- or overwritten by -- another revision of
    the same repo that some other task cached under the bare name.
    """
    slug = (cache_as or repo).replace("/", "__")
    if revision:
        slug = f"{slug}@{revision[:12]}"
    return root_for(repo_type) / slug


def fetch_hf(args: argparse.Namespace) -> Path:
    cache = repo_cache_dir(
        args.repo, args.type, getattr(args, "cache_as", None), getattr(args, "revision", None)
    )

    if cache.is_dir():
        print(f"[fetch] cache hit {args.repo} -> {cache}")
    else:
        print(f"[fetch] downloading {args.repo} -> {cache}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FetchError(
                f"huggingface_hub is required to fetch {args.repo!r}, and the active "
                f"environment does not have it. Add it to the environment's "
                f"docker/images/*.Dockerfile, or pre-populate {cache}."
            ) from exc
        # Final, mechanism-independent xet kill-switch, applied after
        # huggingface_hub (and its file_download submodule) are imported; see the
        # module-load guards above for why both layers are needed.
        _force_disable_xet()
        # Download into a sibling .partial dir and rename it into place only on
        # success: a dir that exists is then always a COMPLETE download, so the
        # `cache.is_dir()` hit above can trust it without per-file checks. An
        # interrupted download leaves only the .partial dir, which the next run
        # clears and retries.
        staging = cache.with_name(cache.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo,
            repo_type=args.type,
            local_dir=str(staging),
            max_workers=args.workers,
            revision=args.revision,
            allow_patterns=list(args.allow) or None,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, cache)

    materialize(cache, args.dest)
    return args.dest


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task_fetch_hf", description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--type",
        choices=("model", "dataset"),
        default="model",
        help="which cache root holds it; selects $AUTORLBENCH_MODELS vs $AUTORLBENCH_DATASETS",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="pin the fetch to a commit; also keys the cache dir so a pinned "
        "download can never collide with another revision's tree",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="GLOB",
        help="restrict the snapshot to these glob patterns (repeatable); "
        "omit to fetch the whole repo",
    )
    parser.add_argument("--dest", required=True, type=Path, help="where the task expects it")
    parser.add_argument(
        "--cache-as",
        metavar="OWNER/NAME",
        default=None,
        help=(
            "cache under this neutral name instead of --repo, for a repo whose "
            "name would reveal the task's answer to the agent. Downloads are "
            "unaffected -- only the cache dir name changes, so bytes still get "
            "reused across runs."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dest = fetch_hf(args)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch] ready {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
