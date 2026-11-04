#!/usr/bin/env python3
"""Turn raw Endless Terminals tasks into a slime prompt set, and validate them.

Endless Terminals distributes each task as a Docker build (`FROM ubuntu:22.04`
plus `apt-get install python3 python3-pip && pip3 install pytest`). The slime
training image already provides Python 3.12 and pytest, and the host has no
route to a package mirror, so instead of building ~2.5k images we transpile
each task's Dockerfile into a plain `setup.sh` whose install commands are
shimmed to no-ops and whose file-creation commands run verbatim.

A task is only admitted to the training set if all four of these hold:

  1. `setup.sh` exits 0 in a fresh tree.
  2. The task's own `environment/test_initial_state.py` PASSES afterwards
     -- i.e. the transpiled setup reproduces the intended initial state.
  3. `tests/test_final_state.py` FAILS on that initial state -- i.e. the reward
     is not already 1 before the agent does anything.
  4. Running the task's reference `solution/solve.sh` makes
     `tests/test_final_state.py` PASS -- i.e. reward 1 is actually reachable.

Tasks that touch absolute paths outside `/home/user` are rejected outright,
because the sandbox only isolates `/home/user` (see et/sandbox.py).

Usage:
    python3 -m et.build_dataset --src /datasets/endless-terminals \
                                --out /tmp/et_prompts
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from et.sandbox import GUEST_HOME, OFFLINE_ENV, check_exec_capable, sync_score  # noqa: E402

DIRECTIVES = {"FROM", "RUN", "ENV", "WORKDIR", "LABEL", "CMD", "ENTRYPOINT", "COPY", "ADD", "USER", "ARG", "EXPOSE"}

HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
ABS_PATH_RE = re.compile(r"(?<![\w$])/(?:[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.*+-]+)*)")

# BuildKit flags on a RUN line -- `RUN --mount=type=cache,target=/root/.cache/pip pip3 install ...`
# and friends. They are Docker syntax, not part of the shell command, so they
# have to come off with the `RUN`: left in, they become the first word of the
# transpiled script (a command named `--mount=...`) and their `target=` paths
# read as writes to /root and /var. ~21% of the published tasks use them.
RUN_FLAGS_RE = re.compile(r"^\s*(?:--[A-Za-z][\w-]*(?:=\S*)?\s+)+")

# Two things that match ABS_PATH_RE without being paths at all. Stripped before
# the scan rather than added to SAFE_PREFIXES, because the point is not that
# they are safe locations -- it is that they are not locations.
URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>`)]*", re.I)  # `//host/a/b` is a URL path
CLOSING_TAG_RE = re.compile(r"</[A-Za-z][\w:-]*\s*>")  # `</body>` is not the directory /body

# Absolute paths a task may mention even though we do not isolate them: they are
# either read-only system locations or already neutralised below.
SAFE_PREFIXES = (
    GUEST_HOME,
    "/dev/null",
    "/dev/urandom",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/etc/passwd",
    "/etc/group",
    "/etc/localtime",
    "/var/lib/apt",
    "/var/cache/apt",
    "/tmp/",
    "/proc/",
)

SHIMS = r"""# --- pip has no index to talk to. Shell functions below cover `pip install`,
# --- but a task may also call a venv's pip by absolute path, which would then
# --- sit in pip's retry loop for minutes. Make any pip fail in under a second.
export PIP_NO_INDEX=1 PIP_RETRIES=0 PIP_TIMEOUT=1 PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# --- network/package commands are unavailable; make them no-ops -------------
apt-get() { :; }
apt() { :; }
apt-key() { :; }
dpkg() { :; }
dpkg-reconfigure() { :; }
pip() { :; }
pip3() { :; }
locale-gen() { :; }
update-locale() { :; }
curl() { echo "build-time network access is disabled" >&2; return 1; }
wget() { echo "build-time network access is disabled" >&2; return 1; }
# --- account management: the image already has a `user`, and the episode's
# --- home directory is created for us, so these would only ever fail --------
useradd() { :; }
adduser() { :; }
usermod() { :; }
groupadd() { :; }
chpasswd() { :; }
# Best-effort: `chown user:user` cannot work without that account, but every
# episode runs as a single uid in its own tree, so ownership is irrelevant.
chown() { command chown "$@" 2>/dev/null || true; }
chgrp() { command chgrp "$@" 2>/dev/null || true; }
export -f apt-get apt apt-key dpkg dpkg-reconfigure pip pip3 locale-gen update-locale curl wget
export -f useradd adduser usermod groupadd chpasswd chown chgrp
"""

# Commands that would mutate the shared container root rather than the episode.
NEUTRALISE = [
    (re.compile(r"rm\s+-rf\s+/var/lib/apt/lists/\*?"), "true"),
    (re.compile(r"rm\s+-rf\s+/var/cache/apt/\S*"), "true"),
]


def parse_instructions(text: str) -> list[tuple[str, str]]:
    """Split a Dockerfile into (DIRECTIVE, full_text) pairs.

    Handles both backslash continuations and Docker heredocs, which is what the
    Endless Terminals Dockerfiles use (1418 of the 2492 published tasks contain
    a heredoc).
    """
    lines = text.split("\n")
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        m = re.match(r"^\s*([A-Za-z]+)\s+(.*)$", raw)
        if not m or m.group(1).upper() not in DIRECTIVES:
            i += 1
            continue
        directive = m.group(1).upper()
        buf = [raw]
        pending = [mm.group(2) for mm in HEREDOC_RE.finditer(raw)]
        cont = raw.rstrip().endswith("\\")
        i += 1
        while (pending or cont) and i < len(lines):
            ln = lines[i]
            buf.append(ln)
            i += 1
            if pending:
                # Inside a heredoc body: only the terminator matters.
                if ln.strip() == pending[0]:
                    pending.pop(0)
                    cont = False
                continue
            pending.extend(mm.group(2) for mm in HEREDOC_RE.finditer(ln))
            cont = ln.rstrip().endswith("\\")
        out.append((directive, "\n".join(buf)))
    return out


def transpile(dockerfile: str) -> str:
    """Dockerfile -> setup.sh body (guest paths kept verbatim)."""
    parts: list[str] = ["#!/bin/bash", "set -euo pipefail", "", SHIMS]
    for directive, block in parse_instructions(dockerfile):
        if directive == "RUN":
            body = re.sub(r"^\s*RUN\s+", "", block, count=1)
            body = RUN_FLAGS_RE.sub("", body)
            for pat, repl in NEUTRALISE:
                body = pat.sub(repl, body)
            parts.append(body)
        elif directive == "ENV":
            body = re.sub(r"^\s*ENV\s+", "", block, count=1).strip()
            if "=" in body:
                parts.append(f"export {body}")
            else:
                k, _, v = body.partition(" ")
                parts.append(f"export {k}={v.strip()!r}")
        elif directive == "WORKDIR":
            path = re.sub(r"^\s*WORKDIR\s+", "", block).strip()
            parts.append(f"mkdir -p {path} && cd {path}")
        # FROM / LABEL / CMD / ENTRYPOINT carry no setup state.
    return "\n\n".join(parts) + "\n"


def unsafe_paths(text: str) -> set[str]:
    text = CLOSING_TAG_RE.sub(" ", URL_RE.sub(" ", text))
    bad = set()
    for m in ABS_PATH_RE.finditer(text):
        p = m.group(0)
        if not any(p == pre.rstrip("/") or p.startswith(pre) for pre in SAFE_PREFIXES):
            bad.add(p)
    return bad


def _env(home: str) -> dict:
    env = dict(
        os.environ,
        HOME=home,
        LC_ALL="C.UTF-8",
        LANG="C.UTF-8",
        DEBIAN_FRONTEND="noninteractive",
        **OFFLINE_ENV,
    )
    for drop in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "CUDA_VISIBLE_DEVICES"):
        env.pop(drop, None)
    return env


def _run(argv, *, cwd, home, timeout, stdin=None):
    """subprocess.run, but a timeout kills the whole process group.

    Plain `subprocess.run(timeout=...)` only kills the direct child, so a
    hung grandchild (classically a `pip install` against an unreachable index)
    survives validation and keeps running for the rest of the build.
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=_env(home),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def run(script: str, cwd: str, home: str, timeout: float = 120.0):
    return _run(["bash", "-s"], cwd=cwd, home=home, timeout=timeout, stdin=script)


def pytest_in(path: str, cwd: str, home: str, timeout: float = 180.0):
    return _run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=cwd,
        home=home,
        timeout=timeout,
    )


def validate(task_src: str, setup_sh: str, scratch: str) -> tuple[bool, str]:
    """The four-way check described in the module docstring."""
    base = tempfile.mkdtemp(prefix="val_", dir=scratch)
    home = os.path.join(base, "home", "user")
    os.makedirs(home, exist_ok=True)
    try:
        # 1. setup must succeed
        r = run(setup_sh.replace(GUEST_HOME, home), cwd=base, home=home)
        if r.returncode != 0:
            return False, f"setup failed rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}"

        # 2. the task's own initial-state test must pass
        init_src = os.path.join(task_src, "environment", "test_initial_state.py")
        if os.path.isfile(init_src):
            dst = os.path.join(base, "test_initial_state.py")
            with open(init_src, encoding="utf-8") as fh:
                body = fh.read().replace(GUEST_HOME, home)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(body)
            r = pytest_in(dst, base, home)
            if r.returncode != 0:
                return False, f"initial-state test failed: {r.stdout[-300:]}"

        # 3. final-state test must NOT already pass
        rc, log = sync_score(os.path.join(task_src, "tests"), base, home)
        if rc == 0:
            return False, "final-state test already passes before the agent acts"

        # 4. the reference solution must make it pass
        solve = os.path.join(task_src, "solution", "solve.sh")
        if not os.path.isfile(solve):
            return False, "no reference solution"
        with open(solve, encoding="utf-8") as fh:
            solve_src = fh.read().replace(GUEST_HOME, home)
        r = run(solve_src, cwd=home, home=home)
        rc, log = sync_score(os.path.join(task_src, "tests"), base, home)
        if rc != 0:
            return False, f"reference solution does not satisfy the verifier: {log[-300:]}"

        return True, "ok"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def evaluate(src_root: str, name: str, scratch: str) -> dict:
    """Admission-test one task directory.

    Depends on nothing but its arguments, and `validate` gives every call its
    own `mkdtemp` sandbox, so this is safe to run concurrently.
    """
    src = os.path.join(src_root, name)
    dockerfile = os.path.join(src, "environment", "Dockerfile")
    instruction = os.path.join(src, "instruction.md")
    final_test = os.path.join(src, "tests", "test_final_state.py")
    if not all(os.path.isfile(p) for p in (dockerfile, instruction, final_test)):
        return {"task": name, "ok": False, "why": "incomplete task directory"}

    with open(dockerfile, encoding="utf-8") as fh:
        setup_sh = transpile(fh.read())

    with open(final_test, encoding="utf-8") as fh:
        test_src = fh.read()
    bad = unsafe_paths(setup_sh) | unsafe_paths(test_src)
    if bad:
        return {"task": name, "ok": False, "why": f"paths outside {GUEST_HOME}: {sorted(bad)[:4]}"}

    try:
        ok, why = validate(src, setup_sh, scratch)
    except subprocess.TimeoutExpired:
        ok, why = False, "timeout during validation"
    except Exception as exc:  # noqa: BLE001 - a broken task must not stop the build
        ok, why = False, f"{type(exc).__name__}: {exc}"
    return {"task": name, "ok": ok, "why": why, "setup_sh": setup_sh}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of raw Endless Terminals task_* dirs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=0, help="stop after N admitted tasks (0 = all)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=min(32, (os.cpu_count() or 4)),
        help="tasks to validate concurrently; each one is a handful of short subprocesses, "
        "so this is wait-bound and may exceed the core count",
    )
    # Must not be under /dev/shm: that is mounted noexec in the training image.
    ap.add_argument("--scratch", default="/tmp/et_validate")
    args = ap.parse_args()

    check_exec_capable(args.scratch)
    tasks_out = os.path.join(args.out, "tasks")
    os.makedirs(tasks_out, exist_ok=True)

    candidates = sorted(
        d for d in os.listdir(args.src) if d.startswith("task_") and os.path.isdir(os.path.join(args.src, d))
    )
    admitted, report = [], []

    # The published Endless Terminals set is ~2.5k tasks, and admitting one runs
    # a setup script, two pytest invocations and a reference solution, so a
    # serial pass takes hours. The tasks are independent and the time goes into
    # waiting on subprocesses, hence threads.
    #
    # Candidates are submitted in sorted order, a chunk at a time, and results
    # are consumed in that same order, so neither the admitted set nor the
    # report depends on --jobs or on which task happens to finish first.
    # --limit stops at the next chunk boundary and then truncates, so it still
    # means "the first N admitted tasks in candidate order".
    chunk = max(1, args.jobs) * 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for start in range(0, len(candidates), chunk):
            batch = candidates[start : start + chunk]
            for rec in pool.map(lambda n: evaluate(args.src, n, args.scratch), batch):
                setup_sh = rec.pop("setup_sh", None)
                name = rec["task"]
                report.append(rec)
                if not rec["ok"]:
                    print(f"  reject {name}: {rec['why']}", flush=True)
                    continue

                src = os.path.join(args.src, name)
                dst = os.path.join(tasks_out, name)
                os.makedirs(dst, exist_ok=True)
                with open(os.path.join(dst, "setup.sh"), "w", encoding="utf-8") as fh:
                    fh.write(setup_sh)
                for rel, out_name in (
                    ("instruction.md", "instruction.md"),
                    ("tests/test_final_state.py", "test_final_state.py"),
                    ("environment/test_initial_state.py", "test_initial_state.py"),
                ):
                    p = os.path.join(src, rel)
                    if os.path.isfile(p):
                        shutil.copyfile(p, os.path.join(dst, out_name))
                admitted.append(name)
                print(f"  admit  {name}", flush=True)
            print(f"  ... {start + len(batch)}/{len(candidates)} checked, {len(admitted)} admitted", flush=True)
            if args.limit and len(admitted) >= args.limit:
                break
    if args.limit:
        admitted = admitted[: args.limit]

    jsonl = os.path.join(args.out, "endless_terminals.jsonl")
    with open(jsonl, "w", encoding="utf-8") as fh:
        for name in admitted:
            with open(os.path.join(tasks_out, name, "instruction.md"), encoding="utf-8") as src_fh:
                fh.write(json.dumps({"prompt": src_fh.read().strip(), "label": name}, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"n_candidates": len(candidates), "n_admitted": len(admitted), "admitted": admitted, "report": report},
            fh,
            indent=2,
        )

    print(f"\nadmitted {len(admitted)} / {len(candidates)} tasks -> {jsonl}")
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
