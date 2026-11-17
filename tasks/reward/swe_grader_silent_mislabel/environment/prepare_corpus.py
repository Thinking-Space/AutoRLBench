"""Materialise the corpus repositories and their pinned environments.

For every repository in `sources_manifest.json`:

  1. verify the shipped source archive against its recorded SHA-256 and unpack it into
     `<workspace>/corpus/repos/<repo_key>`,
  2. make it a git repository with one commit -- the grading pipeline stages each record by
     copying this snapshot and applying patches to it,
  3. create `<workspace>/.venvs/<repo_key>` and install the pinned requirements plus the
     repository itself,
  4. check that the pristine snapshot's test suite is green in that environment. A corpus whose
     baseline is not green cannot be graded, and failing here is cheaper than failing later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=None if cwd is None else str(cwd),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def check(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"prepare: {what} failed ({result.returncode})\n{result.stdout[-2000:]}")


def assert_no_escaping_symlinks(root: Path, workspace: Path, what: str) -> None:
    """No link may point outside the workspace: the harness digests the staged tree and
    resolves every path in it, so a `bin/python` pointing at the system interpreter aborts the
    run after the experiment has already finished."""
    escaping = []
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            path.resolve().relative_to(workspace.resolve())
        except ValueError:
            escaping.append(f"{path} -> {path.resolve()}")
    if escaping:
        raise SystemExit(f"prepare: {what} links outside the workspace: {escaping[:5]}")


def unpack(archive: Path, expected: str, destination: Path) -> None:
    actual = sha256(archive)
    if actual != expected:
        raise SystemExit(f"prepare: {archive.name} digest mismatch: {actual} != {expected}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(destination)


def git_snapshot(repo: Path) -> None:
    check(run(["git", "init", "-q"], repo), f"git init {repo.name}")
    check(run(["git", "add", "-A"], repo), f"git add {repo.name}")
    check(run(["git", "-c", "user.email=corpus@autorlbench", "-c", "user.name=corpus",
               "commit", "-qm", "corpus snapshot"], repo), f"git commit {repo.name}")


def build_environment(python: str, venv: Path, pins: Path, repo: Path) -> None:
    if venv.exists():
        shutil.rmtree(venv)
    # --copies, not symlinks: the harness digests the workspace tree and a `bin/python` that
    # points at the system interpreter resolves outside it.
    check(run([python, "-m", "venv", "--copies", str(venv)]), f"venv {venv.name}")
    pip = str(venv / "bin" / "pip")
    check(run([pip, "install", "-q", "--upgrade", "pip"]), "pip upgrade")
    check(run([pip, "install", "-q", "--no-deps", "-r", str(pins)]), f"pins {venv.name}")
    check(run([pip, "install", "-q", "--no-deps", "-e", "."], repo), f"editable install {venv.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    environment = Path(args.environment).resolve()
    workspace = Path(args.workspace).resolve()
    manifest = json.loads((environment / "sources_manifest.json").read_text())

    repos_root = workspace / "corpus" / "repos"
    repos_root.mkdir(parents=True, exist_ok=True)
    venvs_root = workspace / ".venvs"
    venvs_root.mkdir(parents=True, exist_ok=True)

    for entry in manifest["repositories"]:
        key = entry["repo_key"]
        started = time.time()
        archive = environment / "sources" / entry["archive"]
        repo = repos_root / key
        unpack(archive, entry["sha256"], repo)
        git_snapshot(repo)
        build_environment(args.python, venvs_root / key, environment / "pins" / entry["pins"], repo)
        python = venvs_root / key / "bin" / "python"
        baseline = run([str(python), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"], repo)
        if baseline.returncode != 0:
            raise SystemExit(
                f"prepare: pristine suite of {key} is not green in its pinned environment\n"
                + baseline.stdout[-2000:]
            )
        assert_no_escaping_symlinks(venvs_root / key, workspace, f".venvs/{key}")
        assert_no_escaping_symlinks(repo, workspace, f"corpus/repos/{key}")
        print(f"prepared {key} in {time.time() - started:.1f}s: {baseline.stdout.strip().splitlines()[-1]}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
