"""AIME input and answer helpers for the paper vLLM evaluator."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INSTRUCTION = "Solve the following AIME problem. Think step by step, explain your reasoning, and finish with the integer answer in \\boxed{}."


def load_aime_records(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read AIME parquet rows without requiring the datasets library."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("AIME evaluation requires pyarrow") from exc
    rows = pq.read_table(path).to_pylist()
    result: List[Dict[str, Any]] = []
    for row in rows[: limit if limit is not None else None]:
        if not isinstance(row, dict) or "problem" not in row or "answer" not in row:
            raise ValueError(f"invalid AIME row in {path}: {row!r}")
        result.append(row)
    return result


def extract_answer(text: str) -> Optional[str]:
    """Extract an AIME integer, preferring boxed/final-answer conventions."""
    if not text:
        return None
    boxed = re.findall("\\\\boxed\\s*\\{\\s*(?:\\\\text\\s*\\{\\s*)?(-?\\d{1,3})", text)
    if boxed:
        return boxed[-1].lstrip("+")
    marked = re.findall(
        "(?:final answer|answer|答案)\\s*(?:is|=|：|:)\\s*[^0-9-]*(-?\\d{1,3})", text, flags=re.IGNORECASE
    )
    if marked:
        return marked[-1].lstrip("+")
    numbers = re.findall("(?<![\\w])-?\\d{1,3}(?![\\w])", text)
    return numbers[-1].lstrip("+") if numbers else None


def answers_equal(prediction: Optional[str], expected: str) -> bool:
    """Compare AIME answers numerically (``025`` and ``25`` are equivalent)."""
    if prediction is None:
        return False
    try:
        return 0 <= int(prediction) <= 999 and int(prediction) == int(str(expected).strip())
    except (TypeError, ValueError):
        return prediction.strip() == str(expected).strip()


def _messages(problem: str, instruction: str = DEFAULT_INSTRUCTION) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{instruction}\n\n{problem}"}]
