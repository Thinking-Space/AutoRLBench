"""Per-episode Endless Terminals sandbox.

Endless Terminals ships one Docker image per task (`FROM ubuntu:22.04` + apt).
Building ~50 images is impossible on an air-gapped host and would also defeat
the "avoid duplicate images" constraint, so each episode instead gets a private
directory tree that stands in for the task's `/home/user`.

The guest path `/home/user` is kept *verbatim* in the prompt, the setup script
and the pytest verifier -- it is rewritten to the episode's private directory
only at exec time, and rewritten back in the observation that is shown to the
model. The model therefore sees an authentic transcript while every episode is
fully isolated from every other one. No privileges (no mount namespace, no
chroot) are required, which matters on a shared host.

Commands run in one long-lived `bash` per episode, so `cd`, shell variables and
`export`s persist across turns exactly as they would in a real terminal.

`scratch_root` must allow execution -- see `check_exec_capable`. In particular
it must not be under `/dev/shm`, which the training image mounts `noexec`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import uuid

# The path every Endless Terminals task (45 of 46) addresses exclusively.
GUEST_HOME = "/home/user"

# There is no package index reachable from inside the sandbox. Without these,
# any `pip install` -- whether the policy types one or a task's setup script
# calls a venv's pip by absolute path -- sits in pip's retry loop for minutes
# and outlives the timeout that killed its parent shell. With them it fails in
# under a second.
OFFLINE_ENV = {
    "PIP_NO_INDEX": "1",
    "PIP_RETRIES": "0",
    "PIP_TIMEOUT": "1",
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
}

DEFAULT_CMD_TIMEOUT = 20.0
DEFAULT_MAX_OBS_CHARS = 2000
SETUP_TIMEOUT = 120.0
SCORE_TIMEOUT = 180.0

# asyncio's StreamReader refuses to hand over a line until the whole line is
# buffered, and raises `ValueError` once one line passes its limit -- the
# default being 64 KiB, which a terminal task reaches just by `cat`-ing a
# minified file or a base64 blob. Raise the ceiling, and treat even the raised
# one as "this output is too long" rather than as an error: a single episode
# printing a long line must never be able to end a training run.
SHELL_STREAM_LIMIT = 4 * 1024 * 1024
# Stop accumulating output past this. The observation handed to the model is
# truncated to max_obs_chars (1200 by default) anyway, so everything beyond
# this point was going to be thrown away -- holding it would only let one
# `yes`-style command grow a rollout worker's heap without bound.
MAX_CAPTURE_CHARS = 1 << 20

# Reaping a killed process must not be able to block forever. asyncio resolves
# `Process.wait()` only once the child has exited *and* every pipe it was given
# has reached EOF, so a background job that inherited the shell's stdout keeps
# that future pending even after the shell itself is dead. Several Endless
# Terminals tasks ask for exactly that -- one instructs the policy to launch
# `sleep 1000` and `yes 'Compiling docs...'` as background jobs -- and it hung a
# whole eval at 1199/1200 episodes for half an hour before it was noticed. The
# process has been SIGKILLed by this point either way; waiting is a courtesy, so
# give up on it quickly and let the object be collected.
REAP_TIMEOUT = 10.0


async def _reap(proc) -> None:
    """Wait for an already-killed process, but never longer than REAP_TIMEOUT."""
    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
        await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT)


def check_exec_capable(root: str) -> None:
    """Fail loudly if `root` sits on a `noexec` filesystem.

    Many Endless Terminals tasks create a script and their verifiers assert
    `os.access(path, os.X_OK)`. On a `noexec` mount that returns False even at
    mode 755, so perfectly good tasks get silently rejected and any episode
    that has to run a script it just wrote fails for no visible reason.
    `/dev/shm` is `noexec` in the training image, which makes this very easy to
    hit, so check once up front instead.
    """
    os.makedirs(root, exist_ok=True)
    probe = os.path.join(root, f".execprobe_{uuid.uuid4().hex[:8]}")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(probe, 0o755)
        if not os.access(probe, os.X_OK):
            raise RuntimeError(
                f"{root} is on a noexec filesystem (a mode-755 file there is not executable). "
                "Point ET_SCRATCH / --scratch at a directory that allows execution, e.g. /tmp."
            )
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} characters omitted]...\n{text[-tail:]}"


class Episode:
    """One task attempt in its own private filesystem tree."""

    def __init__(
        self,
        task_dir: str,
        scratch_root: str,
        cmd_timeout: float = DEFAULT_CMD_TIMEOUT,
        max_obs_chars: int = DEFAULT_MAX_OBS_CHARS,
    ) -> None:
        self.task_dir = task_dir
        self.task_id = os.path.basename(task_dir.rstrip("/"))
        self.cmd_timeout = cmd_timeout
        self.max_obs_chars = max_obs_chars

        self.uid = uuid.uuid4().hex[:12]
        self.base = os.path.join(scratch_root, f"ep_{self.task_id}_{self.uid}")
        self.home = os.path.join(self.base, "home", "user")
        # Marker is per-episode so a command that echoes it cannot spoof us.
        self.mark = f"__ET_DONE_{self.uid}__"

        self._shell: asyncio.subprocess.Process | None = None
        self.setup_ok = False
        self.setup_log = ""

    # ---------------------------------------------------------------- paths
    def to_host(self, text: str) -> str:
        """Rewrite guest-visible `/home/user` paths to this episode's tree."""
        return text.replace(GUEST_HOME, self.home)

    def to_guest(self, text: str) -> str:
        """Rewrite host paths back to `/home/user` for display to the model."""
        return text.replace(self.home, GUEST_HOME)

    # ---------------------------------------------------------------- setup
    async def start(self) -> bool:
        os.makedirs(self.home, exist_ok=True)
        setup = os.path.join(self.task_dir, "setup.sh")
        if os.path.isfile(setup):
            with open(setup, encoding="utf-8") as fh:
                script = self.to_host(fh.read())
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-s",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.base,
                env=self._env(),
            )
            try:
                out, _ = await asyncio.wait_for(
                    proc.communicate(script.encode()), timeout=SETUP_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await _reap(proc)
                self.setup_log = "setup timed out"
                return False
            self.setup_log = out.decode("utf-8", "replace")
            if proc.returncode != 0:
                return False

        self.setup_ok = True
        await self._spawn_shell()
        return True

    def _env(self) -> dict:
        env = dict(os.environ)
        env.update(
            {
                "HOME": self.home,
                "PWD": self.home,
                "TERM": "dumb",
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
                "PS1": "",
                "PS2": "",
                "DEBIAN_FRONTEND": "noninteractive",
                **OFFLINE_ENV,
            }
        )
        # Never let an episode inherit the trainer's proxy / CUDA context.
        for drop in ("CUDA_VISIBLE_DEVICES", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(drop, None)
        return env

    async def _spawn_shell(self) -> None:
        # Own session/process group, so killing a wedged turn takes the whole
        # command tree with it rather than orphaning its children. Over a
        # training run that is the difference between a clean host and
        # thousands of leaked background processes.
        self._shell = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.home,
            env=self._env(),
            start_new_session=True,
            limit=SHELL_STREAM_LIMIT,
        )

    async def _kill_shell(self) -> None:
        if self._shell is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(self._shell.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            self._shell.kill()
        await _reap(self._shell)
        self._shell = None

    # ----------------------------------------------------------------- exec
    async def exec(self, command: str) -> str:
        """Run one shell command; return the observation string verbatim in the
        format the Endless Terminals harness uses."""
        host_cmd = self.to_host(command)

        # A command with unbalanced quotes would hang the persistent shell
        # waiting for PS2 continuation, so reject it up front the way the
        # upstream harness effectively does.
        syntax = await asyncio.create_subprocess_exec(
            "bash",
            "-n",
            "-c",
            host_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        syn_out, _ = await syntax.communicate()
        if syntax.returncode != 0:
            return self._format(self.to_guest(syn_out.decode("utf-8", "replace")), 2)

        if self._shell is None or self._shell.returncode is not None:
            await self._spawn_shell()

        assert self._shell is not None and self._shell.stdin is not None
        payload = f"{host_cmd}\nprintf '\\n%s%d\\n' '{self.mark}' \"$?\"\n"
        try:
            self._shell.stdin.write(payload.encode())
            await self._shell.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            await self._kill_shell()
            return self._format("shell died", 1)

        try:
            out, rc = await asyncio.wait_for(self._read_until_mark(), timeout=self.cmd_timeout)
        except asyncio.TimeoutError:
            await self._kill_shell()
            await self._spawn_shell()
            return self._format(f"command timed out after {self.cmd_timeout:.0f}s", 124)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad
            # The shell is in an unknown state, so replace it, and hand the
            # policy an ordinary failed-command observation. Letting this
            # propagate would abort the whole rollout: the eval driver awaits
            # every episode through `asyncio.as_completed`, so one raising
            # episode discards the other several hundred that succeeded.
            await self._kill_shell()
            await self._spawn_shell()
            return self._format(f"could not read command output: {type(exc).__name__}: {exc}", 1)

        return self._format(self.to_guest(out), rc)

    async def _read_chunk(self) -> bytes:
        """`readline`, except an over-long line comes back in pieces.

        asyncio's StreamReader will not hand over a line until it has the whole
        thing buffered, and raises once one line passes its limit. A terminal
        task reaches any limit you pick by `cat`-ing a minified file or a
        base64 blob -- and the raised exception used to propagate all the way
        out of `RolloutManager.eval`, so one task printing one long line ended
        the entire run. The buffered data is still readable, so draining it
        with a bounded `read` both recovers the stream and keeps the output.
        """
        assert self._shell is not None and self._shell.stdout is not None
        try:
            return await self._shell.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError):
            return await self._shell.stdout.read(SHELL_STREAM_LIMIT // 2)

    async def _read_until_mark(self) -> tuple[str, int]:
        assert self._shell is not None and self._shell.stdout is not None
        chunks: list[str] = []
        captured = 0
        # `_read_chunk` does not promise to return whole lines, so the split
        # into lines happens here rather than being assumed. Doing it this way
        # also means the marker is still recognised when it arrives in the same
        # chunk as the tail of a long line.
        pending = ""
        while True:
            data = await self._read_chunk()
            if not data:  # shell exited (e.g. the model ran `exit`)
                chunks.append(pending)
                return "".join(chunks), 1
            pending += data.decode("utf-8", "replace")
            while (nl := pending.find("\n")) != -1:
                line, pending = pending[: nl + 1], pending[nl + 1 :]
                if line.startswith(self.mark):
                    try:
                        rc = int(line[len(self.mark) :].strip() or 1)
                    except ValueError:
                        rc = 1
                    # Drop the single blank line our printf prepended.
                    body = "".join(chunks)
                    if body.endswith("\n"):
                        body = body[:-1]
                    return body, rc
                # Past the cap, keep reading -- the marker is the only way to
                # know the command finished -- but stop accumulating. The
                # observation is truncated to max_obs_chars regardless, so the
                # dropped text could not have been shown anyway.
                if captured < MAX_CAPTURE_CHARS:
                    chunks.append(line)
                    captured += len(line)
            # Bound the live buffer too, not only what is kept: `yes | tr -d
            # '\n'` emits no newline at all, so `pending` would otherwise grow
            # for the whole 20s turn. What is left here is by construction the
            # start of a line with no newline in it, so the marker -- which is
            # always the first thing on its line -- can only be at index 0:
            # keeping the head preserves detection, and the tail dropped is
            # mid-line text well past what the observation can show.
            if len(pending) > MAX_CAPTURE_CHARS:
                pending = pending[:MAX_CAPTURE_CHARS]

    def _format(self, output: str, rc: int) -> str:
        head = "Command executed successfully." if rc == 0 else "Command failed."
        return f"{head} Output: {_truncate(output, self.max_obs_chars)}\n\n(exit_code={rc})"

    # ---------------------------------------------------------------- reward
    async def score(self) -> tuple[float, str]:
        """Run the task's own pytest verifier. Returns (reward, log)."""
        src = os.path.join(self.task_dir, "test_final_state.py")
        if not os.path.isfile(src):
            return 0.0, "no verifier"
        with open(src, encoding="utf-8") as fh:
            test_src = self.to_host(fh.read())
        dst = os.path.join(self.base, "test_final_state.py")
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(test_src)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            dst,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.base,
            env=self._env(),
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=SCORE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await _reap(proc)
            return 0.0, "verifier timed out"
        log = self.to_guest(out.decode("utf-8", "replace"))
        return (1.0 if proc.returncode == 0 else 0.0), log

    # --------------------------------------------------------------- cleanup
    async def close(self) -> None:
        await self._kill_shell()
        shutil.rmtree(self.base, ignore_errors=True)


def sync_score(task_dir: str, base: str, home: str) -> tuple[int, str]:
    """Blocking variant of `score`, used by the offline dataset validator."""
    src = os.path.join(task_dir, "test_final_state.py")
    with open(src, encoding="utf-8") as fh:
        test_src = fh.read().replace(GUEST_HOME, home)
    dst = os.path.join(base, "test_final_state.py")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(test_src)
    env = dict(os.environ, HOME=home, LC_ALL="C.UTF-8", LANG="C.UTF-8", **OFFLINE_ENV)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", dst, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=base,
        env=env,
        capture_output=True,
        text=True,
        timeout=SCORE_TIMEOUT,
    )
    return proc.returncode, proc.stdout + proc.stderr
