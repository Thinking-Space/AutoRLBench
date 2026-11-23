"""Deterministic exact-arithmetic Countdown reward for JustRL-II / Miles."""
from __future__ import annotations

import ast
import json
import re
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
ALLOWED_CHARS_RE = re.compile(r"[0-9+\-*/()\s]+")


def _eval_ast(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, used)
        right = _eval_ast(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError("unsupported binary op")
    raise ValueError("disallowed ast node")


def score_countdown(solution_str: str, ground_truth) -> float:
    if not isinstance(solution_str, str) or ground_truth is None:
        return 0.0
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    elif isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        return 0.0
    matches = ANSWER_RE.findall(solution_str)
    if not matches:
        return 0.0
    expr = matches[-1].strip()
    expr = (
        expr.replace("\\times", "*")
        .replace("\\div", "/")
        .replace("\u00d7", "*")
        .replace("\u00f7", "/")
    )
    if "=" in expr:
        expr = expr.split("=")[0].strip()
    if not expr or len(expr) > 256 or not ALLOWED_CHARS_RE.fullmatch(expr):
        return 0.0
    try:
        tree = ast.parse(expr, mode="eval")
        used: list[int] = []
        val = _eval_ast(tree, used)
    except (SyntaxError, ZeroDivisionError, ValueError, RecursionError):
        return 0.0
    expected_nums = sorted(int(x) for x in gt["numbers"])
    if sorted(used) != expected_nums:
        return 0.0
    return 1.0 if val == Fraction(int(gt["target"])) else 0.0


async def compute_reward(args, sample, **kwargs) -> float:
    """Async custom reward entry point for Miles (--custom-rm-path)."""
    return float(score_countdown(getattr(sample, "response", ""), getattr(sample, "label", None)))
