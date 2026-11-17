"""Tools exposed to the agent for the public repository-debugging workload.

The functions operate on the in-memory project carried in each sample's
metadata, so a tool call is deterministic and needs no network.  Outputs are
sized like real tool observations: whole files, grep hits, pytest summaries.
"""
from __future__ import annotations

import json
import re

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List every file in the repository.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Return the full contents of a file with line numbers.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search all files for a regular expression and return matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the repository test-suite (pytest -q) and return the summary.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "submit_fix", "description": "Submit the diagnosis: defective file, function and explanation.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "function": {"type": "string"}, "explanation": {"type": "string"}},
            "required": ["path", "function"]}}},
]
FINAL_TOOL = "submit_fix"
MAX_OBSERVATION_CHARS = 12000


def _numbered(text: str) -> str:
    return "\n".join(f"{i:4d}  {line}" for i, line in enumerate(text.splitlines(), start=1))


def execute(files: dict[str, str], test_failure: str, name: str, params: dict) -> str:
    if name == "list_files":
        return "\n".join(sorted(files))
    if name == "read_file":
        path = str(params.get("path", "")).lstrip("./")
        if path not in files:
            return f"error: no such file: {path!r}. Use list_files to see available paths."
        return _numbered(files[path])[:MAX_OBSERVATION_CHARS]
    if name == "search":
        try:
            pattern = re.compile(str(params.get("pattern", "")))
        except re.error as exc:
            return f"error: invalid pattern: {exc}"
        hits = [f"{path}:{no}: {line}" for path, text in sorted(files.items())
                for no, line in enumerate(text.splitlines(), start=1) if pattern.search(line)]
        return "\n".join(hits[:200]) or "no matches"
    if name == "run_tests":
        count = sum(1 for p in files if p.startswith("tests/"))
        return f"{test_failure}\n========== 1 failed, {max(0, count * 3 - 1)} passed in 0.41s =========="
    if name == FINAL_TOOL:
        return "submission recorded: " + json.dumps({k: params.get(k) for k in ("path", "function")}, sort_keys=True)
    return f"error: unknown tool {name!r}"


def score(label: str, submission: dict | None) -> float:
    if not submission:
        return 0.0
    expected = json.loads(label)
    path_ok = str(submission.get("path", "")).lstrip("./") == expected["path"]
    fn_ok = str(submission.get("function", "")).strip() == expected["function"]
    return 1.0 if path_ok and fn_ok else (0.2 if path_ok else 0.0)
