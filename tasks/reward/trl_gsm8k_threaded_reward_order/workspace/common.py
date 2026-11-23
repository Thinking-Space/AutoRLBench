"""Prompt format and answer checking shared by training and evaluation."""
import re

SYSTEM = ("You are a careful math tutor. Solve the problem step by step. "
          "End your response with the final numeric answer on its own line, in the form '#### <number>'.")
FINAL = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")


def normalize(x: str) -> str:
    x = x.replace(",", "").strip().rstrip(".")
    try:
        v = float(x)
    except ValueError:
        return x
    return str(int(v)) if v == int(v) else str(v)


def gold_answer(solution: str) -> str:
    return normalize(solution.split("####")[-1])


def extract(text: str):
    """Answer on the last '####' line; None if absent."""
    found = FINAL.findall(text)
    return normalize(found[-1]) if found else None


def messages(question: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
