#!/usr/bin/env python3
"""Load and score Minerva Math and OlympiadBench with SimpleRL's evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Optional


DEFAULT_INSTRUCTION = (
    "Solve the following math problem. Think step by step, explain your reasoning, "
    "and put your final answer within \\boxed{}."
)
DEFAULT_SIMPLELR_ROOT = Path(
    "/historical/shared/qat/vendor/simpleRL-reason/examples/simplelr_math_eval"
)
DEFAULT_SIMPLELR_DEPS = Path("/historical/shared/qat/envs/simplelr-eval-pylibs")
SIMPLELR_NAMES = {
    "amc23": "amc23",
    "minerva_math": "minerva_math",
    "olympiad_bench": "olympiadbench",
}


def _simplelr_modules():
    root = Path(os.environ.get("SIMPLELR_MATH_EVAL_ROOT", DEFAULT_SIMPLELR_ROOT))
    deps = Path(os.environ.get("SIMPLELR_MATH_EVAL_DEPS", DEFAULT_SIMPLELR_DEPS))
    for path in (deps, root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        from grader import math_equal
        from parser import extract_answer, parse_ground_truth, parse_question
    except ImportError as exc:
        raise RuntimeError(
            f"SimpleRL math evaluator is unavailable under {root} (deps: {deps})"
        ) from exc
    return extract_answer, parse_ground_truth, parse_question, math_equal


def load_math_records(
    path: Path, benchmark: str, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Load a full SimpleRL JSONL split and normalize its question/answer fields."""

    if benchmark not in SIMPLELR_NAMES:
        raise ValueError(f"unsupported math benchmark: {benchmark}")
    _, parse_ground_truth, parse_question, _ = _simplelr_modules()
    data_name = SIMPLELR_NAMES[benchmark]
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = parse_question(row, data_name)
            _, answer = parse_ground_truth(row, data_name)
            if not question or answer is None:
                raise ValueError(f"invalid {benchmark} row {line_number} in {path}")
            normalized = dict(row)
            normalized["problem"] = question
            normalized["answer"] = str(answer)
            normalized.setdefault("id", normalized.get("idx", line_number - 1))
            records.append(normalized)
            if limit is not None and len(records) >= limit:
                break
    return records


def extract_math_answer(text: str, benchmark: str) -> Optional[str]:
    if benchmark not in SIMPLELR_NAMES:
        raise ValueError(f"unsupported math benchmark: {benchmark}")
    extract_answer, _, _, _ = _simplelr_modules()
    answer = extract_answer(text, SIMPLELR_NAMES[benchmark])
    return str(answer) if answer not in (None, "") else None


def math_answers_equal(prediction: Optional[str], expected: str) -> bool:
    if prediction is None:
        return False
    _, _, _, math_equal = _simplelr_modules()
    return bool(math_equal(prediction, expected, timeout=True))
