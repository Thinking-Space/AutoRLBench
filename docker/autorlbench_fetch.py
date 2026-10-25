#!/usr/bin/env python3
"""Fetch model weights and datasets from inside the prepare stage.

Installed into the image as `autorlbench_fetch_hf` and `autorlbench_fetch_url`, and
called by each task's `environment/prepare.sh`. This is the in-container
counterpart of `autorlbench/assets/resolver.py` and keeps its two load-bearing
rules:

* **Cache first.** A repo+revision already in `$AUTORLBENCH_MODELS` (models) or
  `$AUTORLBENCH_DATASETS` (datasets) is reused, so the second task that wants
  Qwen2.5-0.5B does not re-download a gigabyte, and a rebuilt image does not
  either. A present cache is *trusted by default*: it was size + sha256 verified
  when it was first downloaded, and re-hashing multi-GB weights on every prepare
  is the dominant cost of a warm-cache run. Pass `--recheck-cache` to re-verify
  on a hit.
* **A checksum mismatch is a hard failure, never a silent re-download.** Every
  fresh download is size + sha256 checked before it is cached, and `--recheck-cache`
  re-applies that check on a hit. Data that does not match what the task recorded
  is either corrupt or altered; turning that into a retry line in a log is how a
  poisoned or truncated weight file reaches a training run nobody is watching.

It is deliberately standalone -- no `autorlbench` import. It runs inside the
evaluation image the task declared (docker/images/<name>.Dockerfile), and that
image contains the task's stack, not the harness.

Usage, as prepare.sh writes it:

    autorlbench_fetch_hf  --repo Qwen/Qwen2.5-0.5B-Instruct \\
                       --revision <40-char-sha> \\
                       --dest "$AUTORLBENCH_PREPARED/models/qwen2.5-0.5b-instruct" \\
                       --expect model.safetensors=<sha256> \\
                       --expect-size model.safetensors=988097824

    autorlbench_fetch_url --url https://example.org/data.tar.gz \\
                       --dest "$AUTORLBENCH_PREPARED/data/gsm8k" \\
                       --sha256 <sha256> --extract
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_CHUNK = 1024 * 1024

# Two cache roots; on the host these are models/ and datasets/ at the repo
# root. The defaults must match MODELS_DIR / DATASETS_DIR in
# autorlbench/core/types.py and docker/autorlbench-activate.sh.
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
# ImportError). This does NOT depend on install-autorlbench-tooling.sh having
# managed to delete the package: on images where hf_xet ships via conda/system
# packaging, `pip uninstall` can't remove it and the file-nuke fallback is
# best-effort, so a surviving hf_xet would otherwise route the download through
# the blocked cas-*.xethub.hf.co CDN (the SSLError seen in prepare). Must run
# before the lazy huggingface_hub import inside fetch_hf. Gated on the same flag
# so an explicit HF_HUB_DISABLE_XET=0 opt-in still gets real xet.
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

    Two roots rather than one with subdirectories, because on the host these are
    two visible, side-by-side directories: forgetting to mount one shows up right
    away as "the download did not persist", instead of silently landing in the
    container's writable layer.
    """
    return DATASETS_ROOT if repo_type == "dataset" else MODELS_ROOT


class FetchError(RuntimeError):
    """The asset is missing, unverifiable, or could not be downloaded."""


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Block-wise, because weight files run to several GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, expect: Dict[str, str], sizes: Dict[str, int]) -> List[str]:
    """Check the declared files under `directory`; return problems, empty if ok.

    Only files the task actually declared are checked. Tokenizer and config
    files are pinned by the revision instead -- hashing every small file in a
    snapshot buys little and makes the task package tedious to write.
    """
    problems: List[str] = []
    for relative in sorted(set(expect) | set(sizes)):
        target = directory / relative
        if not target.exists():
            problems.append(f"{relative}: missing")
            continue
        if not target.is_file():
            problems.append(f"{relative}: expected a file")
            continue
        expected_size = sizes.get(relative)
        if expected_size is not None:
            actual_size = target.stat().st_size
            if actual_size != expected_size:
                # Size first: it is free, and a truncated download -- by far the
                # most common failure -- is far more legible as a size than as a
                # pair of hex strings.
                problems.append(f"{relative}: size {actual_size} != expected {expected_size}")
                continue
        expected_hash = expect.get(relative)
        if expected_hash:
            actual = sha256_file(target)
            if actual.lower() != expected_hash.lower():
                problems.append(
                    f"{relative}: sha256 mismatch\n"
                    f"      expected {expected_hash}\n"
                    f"      actual   {actual}"
                )
    return problems


def require_verified(directory: Path, expect: Dict[str, str], sizes: Dict[str, int], what: str) -> None:
    problems = verify(directory, expect, sizes)
    if problems:
        raise FetchError(f"{what} at {directory} failed verification:\n    " + "\n    ".join(problems))


def presence_problems(directory: Path, expect: Dict[str, str], sizes: Dict[str, int]) -> List[str]:
    """Like ``verify`` but existence (+ declared size) only -- no sha256.

    Cheap enough to run on EVERY cache hit, so a snapshot dir that exists but is
    incomplete is never trusted. That is exactly how an interrupted
    ``snapshot_download`` leaves things: the ``snapshots/<rev>/`` tree and its
    subdirs are created up front, but a file whose blob is still
    ``blobs/<sha>.incomplete`` has no symlink yet, so the declared path does not
    resolve. The old "trust the dir exists" check then materialized an empty
    tree and the caller (e.g. ``tar``) failed downstream with a confusing
    "No such file". Checking presence here turns that into a clean re-download.
    """
    problems: List[str] = []
    for relative in sorted(set(expect) | set(sizes)):
        target = directory / relative
        if not target.exists():  # follows symlinks: a dangling .incomplete fails here
            problems.append(f"{relative}: missing")
            continue
        if not target.is_file():
            problems.append(f"{relative}: expected a file")
            continue
        expected_size = sizes.get(relative)
        if expected_size is not None:
            actual_size = target.stat().st_size
            if actual_size != expected_size:
                problems.append(f"{relative}: size {actual_size} != expected {expected_size}")
    return problems


# --------------------------------------------------------------------------
# materializing cache -> destination
# --------------------------------------------------------------------------


def materialize(source: Path, dest: Path) -> None:
    """Put the cached tree at `dest`, without paying for a second copy if we can.

    Hard links, falling back to a copy across filesystems. The cache roots and
    `$AUTORLBENCH_PREPARED` are normally two directories on one mount, so the link
    path is the usual one and a 1 GB checkpoint costs inodes rather than
    gigabytes.

    Not symlinks: the link target is a container path (/autorlbench/models/...),
    and on the harbor path `--dest` is uploaded into Harbor's container as a
    plain copy of the workspace, where a symlink would dangle or point at
    whatever that container happens to mount there. `--dest` has to be real
    bytes the agent and verifier stages can read on their own.

    `source` is a HuggingFace snapshot dir whose entries are themselves
    symlinks into the sibling blobs store (relative targets like
    ``../../blobs/<sha>``). We resolve each to its real blob before hardlinking:
    ``os.link`` does not follow the symlink on every filesystem (notably the
    some network-mounted home directories), and hardlinking the *symlink* leaves a dangling
    entry at `dest` whose relative target does not resolve from the new
    location. That dangle surfaces far downstream -- an empty ``config.json``
    ("Unrecognized model ... should have a model_type key"), an unreadable
    ``tokenizer.json``, a ``tar: No such file`` -- long after fetch has printed
    "ready". Resolving to the real blob keeps `dest` real bytes everywhere.
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


def hf_repo_dir(repo: str, repo_type: str) -> Path:
    prefix = "datasets--" if repo_type == "dataset" else "models--"
    slug = prefix + repo.replace("/", "--")
    return root_for(repo_type) / "hf" / "hub" / slug


def hf_cache_dir(repo: str, revision: str, repo_type: str) -> Path:
    return hf_repo_dir(repo, repo_type) / "snapshots" / revision


def find_local_blobs(
    repo_type: str, expect: Dict[str, str], sizes: Dict[str, int]
) -> Dict[str, Path]:
    """Locate byte-identical blobs already in the shared cache for sha-pinned files.

    Content-addressed reuse. huggingface_hub stores each file's bytes once under
    ``<root>/hf/hub/<repo-slug>/blobs/<etag>``, where ``<etag>`` is the file's
    content hash (the LFS sha256 oid, and -- confirmed against the resolve
    response's ``x-linked-etag`` -- the xet file's sha256 too). ``--expect
    <file>=<sha256>`` is that same hash, so a blob named ``<sha256>`` under ANY
    repo slug is, by construction, the same bytes.

    This is the ONLY way to satisfy an xet-stored, sha-pinned weight on a cluster
    whose egress proxy blocks the xet CDN (``cas-*.xethub.hf.co`` -- not on the
    whitelist, so every resolve 302 into it dies with an SSL EOF): reuse an
    identical blob that a sibling task's earlier download already left on the
    shared ``/user`` cache. The candidate is matched by name (= its own hash) and
    declared size here, then re-hashed by the caller's ``require_verified`` before
    anything trusts it -- a wrong pick is a hard failure, never silent reuse.
    """
    found: Dict[str, Path] = {}
    hub = root_for(repo_type) / "hf" / "hub"
    if not hub.is_dir():
        return found
    for relative, sha in expect.items():
        for blob in hub.glob(f"*/blobs/{sha.lower()}"):
            if not blob.is_file():
                continue
            want = sizes.get(relative)
            if want is not None and blob.stat().st_size != want:
                continue
            found[relative] = blob
            break
    return found


def seed_hf_cache(repo: str, revision: str, repo_type: str, local_blobs: Dict[str, Path]) -> None:
    """Pre-build this repo's cache pointer from an identical blob already on disk.

    Lays down exactly what a successful ``snapshot_download`` of this file would
    have left -- ``blobs/<etag>`` plus the ``snapshots/<rev>/<file>`` symlink into
    it -- so the next ``snapshot_download`` sees the pointer, returns it, and never
    reaches for the file over the blocked xet CDN. HF keys the blob by the file
    etag, which for these files equals the sha256 we matched on (the blob's own
    filename), so the pointer we build is byte-for-byte the one HF would build.
    The small, non-xet files in the same repo still download normally from the
    mirror; only the sha-pinned blobs we have locally are short-circuited.
    """
    base = hf_repo_dir(repo, repo_type)
    blobs = base / "blobs"
    snap = base / "snapshots" / revision
    for relative, src in local_blobs.items():
        etag = src.name  # blob filename == etag == sha256
        blobs.mkdir(parents=True, exist_ok=True)
        blob_dst = blobs / etag
        if not blob_dst.exists():
            try:
                os.link(os.path.realpath(src), blob_dst)
            except OSError:
                shutil.copyfile(src, blob_dst)
        link = snap / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob_dst, link.parent))
        print(
            f"[fetch] seeded {relative} from byte-identical cached blob {src} "
            f"(xet CDN unreachable; reusing content-addressed cache)"
        )


def fetch_hf(args: argparse.Namespace) -> Path:
    expect, sizes = parse_expectations(args)
    repo_type = args.type
    cached = hf_cache_dir(args.repo, args.revision, repo_type)

    # A snapshot dir that EXISTS is not the same as a snapshot that is COMPLETE:
    # an interrupted snapshot_download leaves the `snapshots/<rev>/` tree (and its
    # subdirs) in place while the file's blob is still `blobs/<sha>.incomplete`
    # with no symlink. So "dir exists" alone must not count as a hit -- we also
    # confirm the declared files actually resolve (cheap: existence + size, no
    # hashing). If they do not, fall through to download, which resumes the
    # .incomplete blob and completes it.
    trusted = False
    if cached.is_dir():
        if args.recheck_cache:
            # Full re-hash on request; still fails hard, never re-downloads, on a
            # genuine content mismatch (a declared file present but wrong bytes).
            require_verified(
                cached, expect, sizes, f"cached {args.repo}@{args.revision[:12]}"
            )
            print(f"[fetch] cache hit (rechecked) {args.repo}@{args.revision[:12]} -> {cached}")
            trusted = True
        else:
            # Default: trust prior sha256 (verified at download time, below), but
            # still cheaply confirm the declared files are present -- this is what
            # catches a half-finished snapshot before materialize copies an empty
            # tree and tar/loader fails far downstream.
            gaps = presence_problems(cached, expect, sizes)
            if gaps:
                print(
                    f"[fetch] cache dir present but incomplete "
                    f"({'; '.join(gaps)}); re-downloading {args.repo}@{args.revision[:12]}"
                )
            else:
                print(f"[fetch] cache hit (trusted) {args.repo}@{args.revision[:12]} -> {cached}")
                trusted = True

    if not trusted:
        print(f"[fetch] downloading {args.repo}@{args.revision[:12]}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FetchError(
                f"huggingface_hub is required to fetch {args.repo!r}, and the active "
                f"environment does not have it. Add it to the environment's "
                f"docker/images/*.Dockerfile, or pre-populate {cached}."
            ) from exc
        # Final, mechanism-independent xet kill-switch, applied after
        # huggingface_hub (and its file_download submodule) are imported. The
        # module-load guards above handle the find_spec / `import hf_xet`
        # detection path, but some huggingface_hub versions decide xet from
        # `importlib.metadata.version("hf_xet")` (dist-info on disk, which the
        # sys.modules sentinel can't hide) and bind `is_xet_available` by value
        # into file_download at import time. Overwrite every such binding to
        # return False so the download can never route through the blocked
        # cas-*.xethub.hf.co CDN, regardless of how this version detects xet.
        _force_disable_xet()
        # cache_dir, not HF_HOME: huggingface_hub freezes HF_HOME into
        # constants.HF_HUB_CACHE at import time, so assigning to os.environ here
        # -- after the import above -- is a no-op, and every download silently
        # went to ~/.cache/huggingface in the container's writable layer instead.
        # The argument is the one entry point that still works post-import.
        #
        # The layout is HF_HOME's own ($HF_HOME/hub), so it *is* the cache layout
        # hf_cache_dir() looks in and the hit above needs no bookkeeping file.
        # Per type, so models land under models/ and datasets under datasets/.
        hub_cache = root_for(repo_type) / "hf" / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        # Before reaching for the network, satisfy any sha-pinned file we already
        # have byte-for-byte elsewhere in the shared cache. On this cluster that
        # is not an optimization but the only working path for xet-stored weights:
        # the xet CDN is firewalled off, so a fresh snapshot_download of such a
        # file dies with an SSL EOF. Seeding the cache pointer from an identical
        # blob turns that file into a cache hit; the repo's other (non-xet) files
        # still download normally from the mirror below.
        local_blobs = find_local_blobs(repo_type, expect, sizes)
        if local_blobs:
            seed_hf_cache(args.repo, args.revision, repo_type, local_blobs)
        downloaded = Path(
            snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                repo_type=repo_type,
                cache_dir=str(hub_cache),
                max_workers=args.workers,
                # None means "the whole repo". Narrowing matters when a repo
                # carries the same weights in several formats: pulling the .bin
                # alongside the .safetensors doubles the download for bytes
                # nothing reads.
                allow_patterns=args.allow or None,
            )
        )
        require_verified(downloaded, expect, sizes, f"downloaded {args.repo}@{args.revision[:12]}")
        cached = downloaded

    materialize(cached, args.dest)
    return args.dest


# --------------------------------------------------------------------------
# plain URL
# --------------------------------------------------------------------------


def fetch_url(args: argparse.Namespace) -> Path:
    expect, sizes = parse_expectations(args)
    if not args.sha256:
        # A URL is mutable in a way a HF revision is not: same address, different
        # bytes tomorrow. Without a checksum there is nothing that would notice.
        raise FetchError("--sha256 is required for --url: an unpinned URL is not reproducible")

    blob_cache = root_for(args.type) / "url" / args.sha256.lower()
    blob = blob_cache / "blob"
    hit = blob.is_file()
    if hit:
        # Trusted by default (see fetch_hf): the blob is keyed by its own sha256
        # and was verified on download, so re-hashing it every run is redundant.
        # --recheck-cache re-hashes on a hit; a mismatch is still a hard failure.
        if args.recheck_cache:
            actual = sha256_file(blob)
            if actual.lower() != args.sha256.lower():
                raise FetchError(
                    f"cached blob for {args.url} is corrupt:\n"
                    f"    expected {args.sha256}\n    actual   {actual}\n"
                    f"    remove {blob} and re-run"
                )
            print(f"[fetch] cache hit (rechecked) {args.url} -> {blob}")
        else:
            print(f"[fetch] cache hit (trusted) {args.url} -> {blob}")
    else:
        print(f"[fetch] downloading {args.url}")
        blob_cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=blob_cache, delete=False) as handle:
            partial = Path(handle.name)
        try:
            urllib.request.urlretrieve(args.url, partial)  # noqa: S310 - URL comes from the task package
            if args.size is not None and partial.stat().st_size != args.size:
                raise FetchError(
                    f"{args.url}: size {partial.stat().st_size} != expected {args.size}"
                )
            actual = sha256_file(partial)
            if actual.lower() != args.sha256.lower():
                raise FetchError(
                    f"{args.url}: sha256 mismatch\n"
                    f"    expected {args.sha256}\n    actual   {actual}"
                )
            # Rename only after it verifies, so an interrupted download can
            # never be mistaken for a cache hit next time.
            partial.rename(blob)
        finally:
            partial.unlink(missing_ok=True)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.extract:
        if args.dest.exists():
            shutil.rmtree(args.dest)
        args.dest.mkdir(parents=True)
        extract(blob, args.dest)
        # Verify the freshly extracted tree after a download or when rechecking;
        # skip it when trusting a cache hit (the blob it came from is trusted).
        if args.recheck_cache or not hit:
            require_verified(args.dest, expect, sizes, f"extracted {args.url}")
    else:
        if args.dest.exists():
            args.dest.unlink()
        try:
            os.link(blob, args.dest)
        except OSError:
            shutil.copyfile(blob, args.dest)
    return args.dest


def extract(blob: Path, target: Path) -> None:
    """Unpack into `target`.

    Member names inside an archive are external input: a member called `../x`
    or `/etc/x` writes outside `target`. tarfile's `filter="data"` exists for
    exactly that (CVE-2007-4559); zip has no equivalent, so it gets a manual
    pass first.
    """
    if tarfile.is_tarfile(blob):
        with tarfile.open(blob) as tar:
            tar.extractall(target, filter="data")
        return
    if zipfile.is_zipfile(blob):
        resolved_target = target.resolve()
        with zipfile.ZipFile(blob) as archive:
            for member in archive.namelist():
                destination = (target / member).resolve()
                if destination != resolved_target and resolved_target not in destination.parents:
                    raise FetchError(f"archive member {member!r} escapes the extraction directory")
            archive.extractall(target)
        return
    raise FetchError(f"unrecognized archive format: {blob}")


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def parse_expectations(args: argparse.Namespace) -> tuple[Dict[str, str], Dict[str, int]]:
    expect: Dict[str, str] = {}
    sizes: Dict[str, int] = {}
    for item in args.expect or []:
        relative, _, value = item.partition("=")
        if not relative or not value:
            raise FetchError(f"--expect wants `<relative-path>=<sha256>`, got {item!r}")
        expect[relative] = value
    for item in args.expect_size or []:
        relative, _, value = item.partition("=")
        if not relative or not value.isdigit():
            raise FetchError(f"--expect-size wants `<relative-path>=<bytes>`, got {item!r}")
        sizes[relative] = int(value)
    return expect, sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autorlbench_fetch", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dest", required=True, type=Path, help="where the task expects it")
        sub.add_argument(
            "--expect",
            action="append",
            metavar="PATH=SHA256",
            help="declared checksum, relative to the fetched tree; repeatable",
        )
        sub.add_argument(
            "--expect-size",
            action="append",
            metavar="PATH=BYTES",
            help="declared size, relative to the fetched tree; repeatable",
        )
        sub.add_argument(
            "--recheck-cache",
            action="store_true",
            help=(
                "re-run the full size + sha256 check on a cache HIT. Off by "
                "default: a present cache was already verified when it was "
                "downloaded, and re-hashing multi-GB weights on every prepare is "
                "the dominant cost of a warm-cache run. A fresh download is "
                "always verified regardless of this flag."
            ),
        )

    hf = subparsers.add_parser("hf", help="fetch a HuggingFace repo at a pinned revision")
    hf.add_argument("--repo", required=True)
    hf.add_argument(
        "--revision",
        required=True,
        help="a full commit sha; a branch or tag name is not a pin",
    )
    hf.add_argument("--type", choices=("model", "dataset"), default="model")
    hf.add_argument(
        "--allow",
        action="append",
        metavar="GLOB",
        help=(
            "restrict the snapshot to these paths; repeatable. Omit to take the "
            "whole repo"
        ),
    )
    hf.add_argument("--workers", type=int, default=8)
    add_common(hf)

    url = subparsers.add_parser("url", help="fetch a file or archive over http(s)")
    url.add_argument("--url", required=True)
    url.add_argument("--sha256", required=True)
    url.add_argument("--size", type=int, default=None)
    url.add_argument(
        "--type",
        choices=("model", "dataset"),
        default="dataset",
        help=(
            "which cache root holds the blob; defaults to dataset because that "
            "is what plain URLs almost always are"
        ),
    )
    url.add_argument("--extract", action="store_true", help="unpack into --dest")
    add_common(url)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "hf":
            if len(args.revision) != 40:
                raise FetchError(
                    f"--revision {args.revision!r} is not a 40-char commit sha; a branch "
                    "or tag can move under the benchmark and change its ground truth"
                )
            dest = fetch_hf(args)
        else:
            dest = fetch_url(args)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch] ready {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
