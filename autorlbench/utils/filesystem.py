from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


def pick_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Return the first existing file among the candidate names."""
    for relative_path in candidates:
        path = root / relative_path
        if path.exists():
            return path
    return None


def ensure_dir(path: Path) -> Path:
    """Create the directory and return the path, for use inside expressions."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_tree(src: Path, dst: Path, ignore: Optional[Iterable[str]] = None) -> Path:
    """Copy a directory tree; if the destination exists, clear it first so the result is deterministic.

    Copies with symlinks=True, preserving symlinks as-is (including dangling ones), matching
    HarborTaskMaterializer._link_tree.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(*(ignore or [])))
    return dst


def iter_files(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> Iterator[Path]:
    """Iterate over files in a directory, optionally skipping generated directories."""
    ignored = set(ignore_dirs or [])
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignored]
        for filename in filenames:
            yield Path(dirpath) / filename


def tree_digest(root: Path, ignore_dirs: Optional[Iterable[str]] = None) -> str:
    """Hash relative paths together with file contents, for change detection and cache keys."""
    digest = hashlib.sha256()
    for path in sorted(iter_files(root, ignore_dirs=ignore_dirs)):
        relative_path = path.resolve().relative_to(root.resolve())
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    return path
