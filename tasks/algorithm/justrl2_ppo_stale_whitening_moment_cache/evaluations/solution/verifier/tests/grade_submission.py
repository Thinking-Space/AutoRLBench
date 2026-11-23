#!/usr/bin/env python3
"""Deterministic Countdown exact-accuracy grader for JustRL-II PPO AutoRLBench tasks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 512 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "expected_train_steps",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "dump_subdir",
    "min_rows",
    "pinned_sha256",
}
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
ALLOWED_CHARS_RE = re.compile(r"[0-9+\-*/()\s]+")

PROMPT_TEMPLATE = (
    "Solve the Countdown arithmetic puzzle. Using the given numbers, form an arithmetic "
    "expression with +, -, *, /, and parentheses that equals the target integer. "
    "Use each given number exactly once. Keep your reasoning brief and output only the "
    "final expression (without '= target') inside <answer>...</answer> tags.\n\n"
    "Example 1:\n"
    "Numbers: [12, 3, 5]\n"
    "Target: 21\n"
    "Reasoning: 12 - 5 = 7, and 3 * 7 = 21.\n"
    "<answer>3 * (12 - 5)</answer>\n\n"
    "Example 2:\n"
    "Numbers: [8, 4, 2, 6]\n"
    "Target: 26\n"
    "Reasoning: 8 * 2 = 16, and 16 + 6 + 4 = 26.\n"
    "<answer>8 * 2 + 6 + 4</answer>\n\n"
    "Example 3:\n"
    "Numbers: [25, 7, 4, 2]\n"
    "Target: 43\n"
    "Reasoning: 7 * 2 = 14, 25 + 14 = 39, and 39 + 4 = 43.\n"
    "<answer>25 + 7 * 2 + 4</answer>\n\n"
    "Now solve:\n"
    "Numbers: {nums}\n"
    "Target: {target}\n"
    "Reasoning:"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_SPEC_KEYS - set(value)
    if missing:
        raise ValueError(f"grader_spec.json missing keys: {sorted(missing)}")
    if int(value["schema_version"]) != 3:
        raise ValueError("grader_spec.json schema_version must be 3")
    value["final_step"] = int(value["final_step"])
    value["expected_train_steps"] = int(value["expected_train_steps"])
    value["min_rows"] = int(value["min_rows"])
    return value


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


def countdown_correct(response: str, gt: Any) -> bool:
    if not isinstance(response, str) or gt is None:
        return False
    try:
        parsed_gt = json.loads(gt) if isinstance(gt, str) else gt
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed_gt, dict) or "numbers" not in parsed_gt or "target" not in parsed_gt:
        return False
    matches = ANSWER_RE.findall(response)
    if not matches:
        return False
    expr = matches[-1].strip()
    expr = expr.replace("\\times", "*").replace("\\div", "/").replace("\u00d7", "*").replace("\u00f7", "/")
    if "=" in expr:
        expr = expr.split("=")[0].strip()
    if not expr or len(expr) > 256 or not ALLOWED_CHARS_RE.fullmatch(expr):
        return False
    try:
        tree = ast.parse(expr, mode="eval")
        used: list[int] = []
        val = _eval_ast(tree, used)
    except (SyntaxError, ZeroDivisionError, ValueError, RecursionError):
        return False
    expected_nums = sorted(int(x) for x in parsed_gt["numbers"])
    if sorted(used) != expected_nums:
        return False
    return val == Fraction(int(parsed_gt["target"]))


RULES = {"countdown": countdown_correct}


def find_dumps(results: Path, roots: list[str], subdir: str, final_step: int) -> list[Path]:
    results_resolved = results.resolve()
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        if not root.resolve().is_relative_to(results_resolved):
            continue
        for path in root.rglob(f"{final_step}.jsonl"):
            if path.is_symlink() or not path.is_file() or path.parent.name != subdir:
                continue
            rel = path.relative_to(root)
            if any((root / Path(*rel.parts[: i + 1])).is_symlink() for i in range(len(rel.parts) - 1)):
                continue
            if not path.resolve().is_relative_to(results_resolved):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)


def score_dump(path: Path, rule: str = "countdown") -> tuple[int, int, int, list[str]]:
    correct_fn = RULES[rule]
    rows = correct = chars = 0
    problems: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {lineno}: not JSON")
                continue
            if not isinstance(row, dict) or "output" not in row or "gts" not in row:
                problems.append(f"line {lineno}: missing output/gts")
                continue
            rows += 1
            correct += int(correct_fn(row["output"], row["gts"]))
            chars += len(row["output"]) if isinstance(row["output"], str) else 0
    return rows, correct, chars, problems


def find_metrics(results: Path, roots: list[str]) -> Path | None:
    results_resolved = results.resolve()
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        if not root.resolve().is_relative_to(results_resolved):
            continue
        direct = root / "metrics.jsonl"
        if direct.is_file() and not direct.is_symlink() and direct.resolve().is_relative_to(results_resolved):
            found.append(direct)
            continue
        for path in root.rglob("metrics.jsonl"):
            rel = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or "wandb" in rel.parts:
                continue
            if any((root / Path(*rel.parts[: i + 1])).is_symlink() for i in range(len(rel.parts) - 1)):
                continue
            if not path.resolve().is_relative_to(results_resolved):
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)[0] if found else None


def training_reality(results: Path, spec: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    final_step = spec["final_step"]
    expected_steps = spec["expected_train_steps"]
    val_key = spec.get("val_key", "eval/countdown")
    num_critic_only = int(spec.get("num_critic_only_steps", 3))
    path = find_metrics(results, spec["log_roots"])
    details: dict[str, Any] = {
        "metrics_path": str(path.relative_to(results)) if path else None,
        "metrics_rows": 0,
        "rollout_steps_continuous": False,
        "actor_steps_logged": 0,
        "final_eval_logged": False,
    }
    if path is None:
        return False, details, (
            f"missing metrics.jsonl under {spec['log_roots']} (the shipped runner sets MILES_METRICS_FILE "
            f"and logs rollout, training, and evaluation steps; a run without it has no step-{final_step} verification)"
        )
    rows: list[dict[str, Any]] = []
    first_problem = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                first_problem = first_problem or f"line {lineno}: not JSON"
                continue
            if isinstance(row, dict):
                rows.append(row)
    if first_problem:
        return False, details, f"{path.relative_to(results)}: malformed training log ({first_problem})"

    details["metrics_rows"] = len(rows)
    rollout_steps = sorted(
        {row["step"] for row in rows if row.get("step_key") == "rollout/step" and isinstance(row.get("step"), int)}
    )
    actor_steps = sorted(
        {
            row["step"]
            for row in rows
            if row.get("step_key") == "train/step"
            and isinstance(row.get("step"), int)
            and isinstance(row.get("data"), dict)
            and "train/pg_loss" in row["data"]
        }
    )
    details["rollout_steps_continuous"] = rollout_steps == list(range(expected_steps))
    details["actor_steps_logged"] = len(actor_steps)
    details["final_eval_logged"] = any(
        row.get("step_key") == "eval/step"
        and row.get("step") == final_step
        and isinstance(row.get("data"), dict)
        and val_key in row["data"]
        for row in rows
    )
    expected_actor_steps = expected_steps - num_critic_only
    if not details["rollout_steps_continuous"]:
        return False, details, (
            f"{path.relative_to(results)} does not show a complete run: expected rollout steps "
            f"0..{expected_steps - 1}, got {len(rollout_steps)} unique steps"
        )
    if len(actor_steps) < expected_actor_steps:
        return False, details, (
            f"{path.relative_to(results)} logged only {len(actor_steps)} actor policy updates "
            f"(expected >= {expected_actor_steps} with train/pg_loss)"
        )
    if not details["final_eval_logged"]:
        return False, details, (
            f"{path.relative_to(results)} has no {val_key} row at eval step {final_step}"
        )
    return True, details, None


def _countdown_jsonl_consistent(workspace: Path, check: dict[str, Any], *, is_val: bool) -> bool:
    try:
        import pyarrow.parquet as pq

        raw_table = pq.read_table(workspace / check["raw"])
        jsonl_path = workspace / check["path"]
        lines = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError, ImportError):
        return False
    expected_rows = int(check["rows"])
    total = raw_table.num_rows
    if len(lines) != expected_rows or total < expected_rows:
        return False
    raw_targets = raw_table["target"].to_pylist()
    raw_nums = raw_table["nums"].to_pylist()
    expected_indices = list(range(total - expected_rows, total)) if is_val else list(range(expected_rows))

    seen: set[int] = set()
    for row in lines:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            return False
        idx = meta.get("index")
        if not isinstance(idx, int) or idx in seen or not (0 <= idx < total):
            return False
        seen.add(idx)
        target = int(raw_targets[idx])
        numbers = [int(x) for x in raw_nums[idx]]
        expected_gt = {"numbers": numbers, "target": target}
        try:
            parsed_gt = json.loads(row.get("label", "")) if isinstance(row.get("label"), str) else row.get("label")
        except (TypeError, json.JSONDecodeError):
            return False
        if parsed_gt != expected_gt:
            return False
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict):
            return False
        if (
            prompt[0].get("role") != "user"
            or prompt[0].get("content") != PROMPT_TEMPLATE.format(nums=numbers, target=target)
        ):
            return False
    return seen == set(expected_indices)


def stage_integrity(spec: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    workspace = os.environ.get("AUTORLBENCH_WORKSPACE", "")
    verifier = os.environ.get("AUTORLBENCH_VERIFIER", "")
    if not workspace or not verifier:
        return None, {}
    details: dict[str, Any] = {}
    runner = spec.get("runner", "run_justrl2_ppo.sh")
    if spec.get("runner_pinned", True):
        try:
            details[runner] = (Path(workspace) / runner).read_bytes() == (
                Path(verifier) / "resources" / runner
            ).read_bytes()
        except OSError:
            details[runner] = False
    for rel, expected in spec.get("pinned_sha256", {}).items():
        try:
            digest = hashlib.sha256((Path(workspace) / rel).read_bytes()).hexdigest()
        except OSError:
            digest = ""
        accepted = expected if isinstance(expected, list) else [expected]
        details[rel] = digest in accepted
    train_check = spec.get("train_jsonl_consistency")
    if train_check:
        details[train_check["path"] + " (matches raw Countdown train slice)"] = _countdown_jsonl_consistent(
            Path(workspace), train_check, is_val=False
        )
    val_check = spec.get("val_jsonl_consistency")
    if val_check:
        details[val_check["path"] + " (matches raw Countdown test slice)"] = _countdown_jsonl_consistent(
            Path(workspace), val_check, is_val=True
        )
    return all(details.values()), details


def grade(spec: dict[str, Any], results: Path):
    final_step = spec["final_step"]
    baseline = Decimal(str(spec["historical_final_baseline"]))
    required = Decimal(str(spec["required_final_score"]))
    required_max = spec.get("required_final_score_max")
    required_max = Decimal(str(required_max)) if required_max is not None else None
    errors: list[str] = []
    dumps = find_dumps(results, spec["log_roots"], spec["dump_subdir"], final_step)
    judged = None
    source = None
    rows = correct = chars = 0
    mean_chars = None
    problems: list[str] = []
    if not dumps:
        errors.append(
            f"missing {spec['dump_subdir']}/{final_step}.jsonl under {spec['log_roots']} "
            "(the run must reach its final evaluation with MILES_EVAL_DUMP_DIR as shipped)"
        )
    else:
        source = dumps[0]
        rows, correct, chars, problems = score_dump(source, spec.get("rule", "countdown"))
        mean_chars = chars / rows if rows else None
        if rows < spec["min_rows"]:
            errors.append(
                f"{source.relative_to(results)} has {rows} scored rows, fewer than the {spec['min_rows']} required"
            )
        else:
            judged = Decimal(correct) / Decimal(rows)
            if judged < required:
                errors.append(
                    f"held-out accuracy ({spec.get('rule', 'countdown')} rule) {judged:.4f} at step {final_step} is below required {required}"
                )
            elif required_max is not None and judged > required_max:
                errors.append(
                    f"held-out accuracy ({spec.get('rule', 'countdown')} rule) {judged:.4f} at step {final_step} is above the maximum {required_max}"
                )
    _, reality_details, reality_error = training_reality(results, spec)
    if reality_error:
        errors.append(reality_error)
    integrity_ok, integrity_details = stage_integrity(spec)
    if integrity_ok is False:
        errors.append(
            "the rerun must use the workspace as shipped apart from the repair: failed workspace checks "
            f"{sorted(k for k, ok in integrity_details.items() if not ok)}"
        )
    passed = not errors
    score = 1.0 if passed else 0.0
    judged_f = float(judged) if judged is not None else None
    metrics = {
        "task_id": spec["task_id"],
        "metric_key": spec["metric_key"],
        "final_step": final_step,
        "score_mode": "final",
        "historical_final_baseline": float(baseline),
        "required_final_score": float(required),
        "required_final_score_max": float(required_max) if required_max is not None else None,
        "best_step": final_step if judged is not None else None,
        "best_score": judged_f,
        "best_source_log": str(source.relative_to(results)) if source else None,
        "final_score": judged_f,
        "absolute_improvement": float(judged - baseline) if judged is not None else None,
        "scored_rows": rows,
        "correct_rows": correct,
        "mean_response_chars": mean_chars,
        "dump_problems": problems[:20],
        "training_reality": reality_details,
        "stage_integrity": integrity_ok if integrity_ok is not None else "not checked (no stage environment)",
        "stage_integrity_files": integrity_details,
        "source_logs": [str(p.relative_to(results)) for p in dumps],
        "errors": errors,
        "passed": passed,
        "score": score,
        "case_score": score,
        "deterministic": True,
        "judge_used": False,
    }
    experiment = {
        "task_id": spec["task_id"],
        "logs": {"source_logs": metrics["source_logs"]},
        "metrics": {
            "metric_key": spec["metric_key"],
            "metric_status": "available" if judged is not None else "missing",
            "best_step": metrics["best_step"],
            "best_score": judged_f,
            "best_source_log": metrics["best_source_log"],
            "scored_rows": rows,
            "correct_rows": correct,
        },
    }
    return experiment, metrics, {"passed": passed, "score": score, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score step-final Countdown generations against grader_spec.json")
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("AUTORLBENCH_RESULTS", "/results")))
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    try:
        spec = load_spec(Path(__file__).resolve().parent.parent / "resources" / "grader_spec.json")
        experiment, metrics, result = grade(spec, results)
        write_json(results / "experiment_metrics.json", experiment)
        write_json(results / "metrics.json", metrics)
        write_json(results / "result.json", result)
        if metrics["errors"]:
            print("; ".join(metrics["errors"]), file=sys.stderr)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metrics = {
            "score": 0.0,
            "case_score": 0.0,
            "passed": False,
            "errors": [f"verifier error: {exc}"],
            "deterministic": True,
            "judge_used": False,
        }
        result = {"passed": False, "score": 0.0, "metrics": metrics}
        try:
            write_json(results / "metrics.json", metrics)
            write_json(results / "result.json", result)
        except OSError:
            pass
        print(metrics["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
