"""Python interpreter tool for the GSM8K tool-use recipe (stateless verl function tool)."""
import os
import subprocess
import sys
import tempfile

from verl.tools.function_tool import function_tool

TIMEOUT_S = float(os.environ.get("PY_TOOL_TIMEOUT_S", "5"))


@function_tool
def python(code: str) -> str:
    """Run a short Python 3 program and return what it prints.

    Args:
        code: The Python source to execute. Use print() to show results.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {TIMEOUT_S:g} s"
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or "(no output)"
