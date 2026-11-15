"""Submitter-only reference repair for the public generation adapter."""
from pathlib import Path
import sys


def apply(workspace):
    path = workspace / "runtime/public_generate.py"
    text = path.read_text()
    before = "    prompt_value = sample.prompt\n"
    after = "    prompt_value = public_messages(sample.label)\n"
    if text.count(before) != 1:
        raise ValueError("Unexpected public adapter source")
    path.write_text(text.replace(before, after, 1))


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
