# Snapshot from QAT/opqad/evaluation/eval_aime.py; avoids importing torch for CPU scoring.
import re
from typing import Optional

def extract_answer(text: str) -> Optional[str]:
    """Extract an AIME integer, preferring boxed/final-answer conventions."""

    if not text:
        return None
    # Models use both LaTex \u005cboxed{123} and ``\\boxed{\\text{123}}``.
    boxed = re.findall(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?(-?\d{1,3})", text)
    if boxed:
        return boxed[-1].lstrip("+")
    marked = re.findall(
        r"(?:final answer|answer|答案)\s*(?:is|=|：|:)\s*[^0-9-]*(-?\d{1,3})",
        text,
        flags=re.IGNORECASE,
    )
    if marked:
        return marked[-1].lstrip("+")
    # Last standalone 0--999 integer is a conservative fallback.  Restricting
    # to three digits avoids selecting years or intermediate large constants.
    numbers = re.findall(r"(?<![\w])-?\d{1,3}(?![\w])", text)
    return numbers[-1].lstrip("+") if numbers else None

def answers_equal(prediction: Optional[str], expected: str) -> bool:
    """Compare AIME answers numerically (``025`` and ``25`` are equivalent)."""

    if prediction is None:
        return False
    try:
        return 0 <= int(prediction) <= 999 and int(prediction) == int(str(expected).strip())
    except (TypeError, ValueError):
        return prediction.strip() == str(expected).strip()
