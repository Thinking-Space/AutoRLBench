"""MBPP reward: 1.0 when the response's code passes every assert of the problem.

Training rollouts go to a long-lived exec server (thousands of samples per step, no interpreter
start-up per sample); validation uses a fresh interpreter per sample, the reference harness.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading

CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
TRAIN_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 10

# Runs in the exec server. Requests and replies travel on private copies of stdin/stdout, so code
# under test that reads input or prints cannot touch the protocol.
_SERVER = r"""
import json, os, sys
proto_in = os.fdopen(os.dup(0), "r")
proto_out = os.fdopen(os.dup(1), "w")
null = os.open(os.devnull, os.O_RDWR)
os.dup2(null, 0); os.dup2(null, 1); os.dup2(null, 2)
sys.stdin = open(os.devnull); sys.stdout = open(os.devnull, "w"); sys.stderr = sys.stdout


def new_namespace():
    return {"__name__": "__mbpp__"}


for line in proto_in:
    req = json.loads(line)
    namespace = new_namespace()
    ok = True
    try:
        exec(compile(req["code"], "<mbpp>", "exec"), namespace)
        exec(compile(req["tests"], "<mbpp-tests>", "exec"), namespace)
    except BaseException:
        ok = False
    proto_out.write(json.dumps({"ok": ok}) + "\n")
    proto_out.flush()
"""


class _ExecServer:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._replies = None

    def _start(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self._replies = queue.Queue()
        proc, replies = self._proc, self._replies

        def pump():
            for line in proc.stdout:
                replies.put(line)
            replies.put(None)

        threading.Thread(target=pump, daemon=True).start()

    def _kill(self):
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def run(self, code: str, tests: str) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            try:
                self._proc.stdin.write(json.dumps({"code": code, "tests": tests}) + "\n")
                self._proc.stdin.flush()
                line = self._replies.get(timeout=TRAIN_TIMEOUT_S)
            except (queue.Empty, BrokenPipeError, OSError):
                line = None
            if line is None:  # timeout or the server died (e.g. os._exit in the code under test)
                self._kill()
                return False
            return bool(json.loads(line)["ok"])


_SERVER_POOL = _ExecServer()


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else ""


def _exec_subprocess(program: str) -> bool:
    try:
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    spec = json.loads(ground_truth)
    code = extract_code(solution_str)
    setup = spec.get("setup") or ""
    tests = "\n".join(spec["tests"])
    if (extra_info or {}).get("split") == "test":
        return float(_exec_subprocess(setup + "\n" + code + "\n" + tests))
    return float(_SERVER_POOL.run(setup + "\n" + code, tests))
