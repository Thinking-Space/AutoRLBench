"""Independent repair workspaces with fixed tests rerun on a clean upstream snapshot.

The tool API exposes only the localized source module. It is not a general-purpose
shell or an OS security boundary; run generated code only in the disposable worker.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


SYSTEM = """You maintain a Python project with a failing regression test. Repair the localized
source module while preserving its other behavior. Use the tools to inspect the source,
make a minimal edit, and run the fixed regression/non-regression test command. You may
make at most six tool calls. Only the named source module can be edited. Tests and runtime
configuration are protected. When the repair is complete, finish with a short explanation.
Do not disable assertions, tests, or test reporting. The reward comes from passing the
original tests on a fresh checkout containing your source edit."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RepairEnvironment:
    def __init__(self, corpus: Path, python: str, output: Path, max_calls=6):
        self.corpus = Path(corpus)
        self.python = python
        self.output = Path(output)
        self.records = {r["case_id"]: r for r in json.loads((self.corpus / "manifest.json").read_text())["cases"]}
        self.max_calls = max_calls
        self.work = None

    def _log(self, event, **data):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "tool_traces.jsonl").open("a") as file:
            file.write(json.dumps({"trajectory_id": self.trajectory_id, "case_id": self.case_id,
                                   "event": event, "call": self.calls, **data}) + "\n")

    def _claim(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise ValueError("The six-call tool budget is exhausted")

    def reset(self, case_id, **kwargs):
        if self.work is not None:
            shutil.rmtree(self.work)
        self.case_id = case_id
        self.record = self.records[case_id]
        self.case = self.corpus / "cases" / case_id
        self.trajectory_id = uuid.uuid4().hex
        self.work = Path(tempfile.mkdtemp(prefix="pc-repair-"))
        self.target = self.work / self.record["source_file"]
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.original = (self.case / "base" / self.record["source_file"]).read_bytes()
        self.target.write_bytes(self.original)
        self.calls = 0
        self._log("reset", source_sha256=digest(self.original), workspace=str(self.work))
        # Identical for all four samples of this case: no rollout ID, nonce, or random observation.
        failure = (self.case / "defective.log").read_text()
        summary = failure[-2200:]
        return f"\nCase: {case_id}\nEditable module: {self.record['source_file']}\nFailing test output:\n{summary}\n"

    def inspect_source(self, start_line: int = 1, end_line: int = 80) -> str:
        """Read a numbered range in the editable source module.

        Args:
            start_line: First line, starting at one.
            end_line: Last line, at most 100 lines after the start.
        """
        self._claim()
        lines = self.target.read_text().splitlines()
        if start_line < 1 or end_line < start_line or end_line - start_line >= 100:
            raise ValueError("Choose a valid range of at most 100 lines")
        result = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines) if start_line <= i+1 <= end_line)
        self._log("inspect_source", start_line=start_line, end_line=end_line, result=result)
        return result

    def search_source(self, text: str) -> str:
        """Find literal text in the editable source module.

        Args:
            text: Literal substring to search for.
        """
        self._claim()
        result = "\n".join(f"{i+1}: {line}" for i, line in enumerate(self.target.read_text().splitlines()) if text in line)
        result = result[:6000]
        self._log("search_source", text=text, result=result)
        return result or "No matching lines"

    def replace_source(self, old: str, new: str) -> str:
        """Replace one unique exact substring in the editable source module.

        Args:
            old: Existing source text that occurs exactly once, without line numbers.
            new: Replacement Python source, with correct indentation.
        """
        self._claim()
        source = self.target.read_text()
        if not old or source.count(old) != 1:
            result = "Edit rejected: old text must occur exactly once. Inspect the current source."
        elif len(new) > 12000:
            result = "Edit rejected: replacement exceeds the size limit."
        else:
            source = source.replace(old, new, 1)
            try:
                ast.parse(source)
            except SyntaxError as exc:
                result = f"Edit rejected: invalid Python syntax: {exc}"
            else:
                self.target.write_text(source)
                result = "Edit applied."
        self._log("replace_source", old=old, new=new, result=result, source_sha256=digest(self.target.read_bytes()))
        return result

    def _test(self):
        source = self.target.read_bytes()
        original_ast, changed_ast = ast.parse(self.original), ast.parse(source)
        if sum(isinstance(n, ast.Assert) for n in ast.walk(changed_ast)) < sum(isinstance(n, ast.Assert) for n in ast.walk(original_ast)):
            return {"passed": False, "reason": "removed_assertion", "output": "Removing assertions is forbidden."}
        # A fresh reference snapshot prevents edits to earlier test artifacts or local caches from counting.
        with tempfile.TemporaryDirectory(prefix="pc-check-") as temporary:
            root = Path(temporary)
            checkout = root / "checkout"
            shutil.copytree(self.case / "base", checkout, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            (checkout / self.record["source_file"]).write_bytes(source)
            report = root / "report.xml"
            env = {"PATH": os.defpath, "HOME": str(root), "TMPDIR": str(root), "TERM": "xterm",
                   "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""}
            command = [self.python, str(Path(__file__).with_name("test_runner.py").resolve()),
                       "--assert=plain", "-q", "-p", "pytest_mock",
                       "-o", "addopts=", f"--junitxml={report}", *self.record["tests"]]
            started = time.monotonic()
            with (root / "test.log").open("w+") as log:
                process = subprocess.Popen(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                timed_out = False
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
                log.seek(0)
                text = log.read(32000).replace(str(checkout), "<checkout>").replace(str(root), "<test-run>")
            counts = {}
            try:
                xml = ET.parse(report).getroot()
                suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
                counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                          for key in ["tests", "failures", "errors", "skipped"]}
            except (OSError, ET.ParseError, ValueError):
                pass
            passed = (returncode == 0 and not timed_out and counts.get("tests") == self.record["expected_tests"]
                      and not any(counts.get(k, 0) for k in ["failures", "errors", "skipped"]))
            return {"passed": passed, "returncode": returncode, "timeout": timed_out,
                    "seconds": time.monotonic()-started, "counts": counts, "output": text[-4500:],
                    "source_sha256": digest(source)}

    def run_tests(self) -> str:
        """Run the fixed regression and non-regression tests on a fresh checkout."""
        self._claim()
        result = self._test()
        self._log("run_tests", **result)
        return result["output"]

    def get_reward(self):
        result = self._test()
        reward = float(result["passed"] and self.calls <= self.max_calls)
        self._log("grade", reward=reward, **result)
        patch_dir = self.output / "submitted_sources"
        patch_dir.mkdir(exist_ok=True)
        (patch_dir / f"{self.trajectory_id}.py").write_bytes(self.target.read_bytes())
        return reward
