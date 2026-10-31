#!/usr/bin/env python3
"""Re-score the fresh run's held-out generation dump with a fixed \boxed{} rule and assign a binary score.

The recipe's logged `val-core/openai/gsm8k/acc/mean@1` is produced by verl 0.9.0's Math-Verify scorer, whose 4-worker
process pool charges queueing time against its 30 s per-sample timeout and silently scores a timed-out sample 0; on
a busy host it under-reports the 1,319-sample validation pass by up to ~0.10 without any message. The grader
therefore does not read logged metrics: it scores the responses the run dumped at `final_step`
(`trainer.validation_data_dir` = `experiment_logs/val_generations`) with a pure-Python rule that is not part of the
workspace: a response is correct iff the number inside its last `\boxed{...}` equals the gold answer (`gts`)
numerically. The rule is deterministic and independent of host load.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 512 * 1024 * 1024
REQUIRED_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "metric_key",
    "final_step",
    "historical_final_baseline",
    "required_final_score",
    "log_roots",
    "dump_subdir",
    "min_rows",
}
LAST_RE = re.compile(r"-?\d+(?:\.\d+)?")
FRAC_RE = re.compile(r"\s*\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}\s*")


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
    value["min_rows"] = int(value["min_rows"])
    return value


def to_number(token: str) -> Decimal | None:
    token = token.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def gold_number(gt: Any) -> Decimal | None:
    if gt is None:
        return None
    if isinstance(gt, dict):
        gt = gt.get("ground_truth", gt)
    text = str(gt)
    direct = to_number(text)
    if direct is not None:
        return direct
    found = LAST_RE.findall(text.replace(",", ""))
    return to_number(found[-1]) if found else None


def last_boxed(text: str) -> str | None:
    """Contents of the last `\boxed{...}` (or `\fbox{...}`) with balanced braces; None if there is none."""
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def boxed_number(box: str) -> Decimal | None:
    """The number stated in a boxed answer: LaTeX wrappers, currency, percent and thousands separators removed,
    simple \frac{a}{b} evaluated, the first number taken (units and words ignored)."""
    s = box.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    for junk in ("\\$", "$", "\\%", "%", "\\!", ",", "\\,"):
        s = s.replace(junk, "")
    s = s.replace("dfrac", "frac")
    frac = FRAC_RE.fullmatch(s)
    if frac:
        num, den = to_number(frac.group(1)), to_number(frac.group(2))
        if num is None or den is None or den == 0:
            return None
        return num / den
    found = LAST_RE.search(s)
    return to_number(found.group(0)) if found else None


def boxed_correct(response: str, gt: Any) -> bool:
    """The number inside the last `\boxed{}` of the response equals the gold answer (numeric comparison, 1e-6 relative)."""
    gold = gold_number(gt)
    if gold is None or not isinstance(response, str):
        return False
    box = last_boxed(response)
    if box is None:
        return False
    value = boxed_number(box)
    if value is None:
        return False
    tol = Decimal("1e-6") * max(Decimal(1), abs(gold))
    return abs(value - gold) <= tol


def find_dumps(results: Path, roots: list[str], subdir: str, final_step: int) -> list[Path]:
    """`<root>/**/<subdir>/<final_step>.jsonl`, regular files only (no symlinks anywhere below the root), newest first."""
    found = []
    for root_name in roots:
        root = results / root_name
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob(f"{final_step}.jsonl"):
            if path.is_symlink() or not path.is_file() or path.parent.name != subdir:
                continue
            rel = path.relative_to(root)
            if any((root / Path(*rel.parts[: i + 1])).is_symlink() for i in range(len(rel.parts) - 1)):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)


def score_dump(path: Path) -> tuple[int, int, list[str]]:
    rows = correct = 0
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
            correct += boxed_correct(row["output"], row["gts"])
    return rows, correct, problems


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
    rows = correct = 0
    problems: list[str] = []
    if not dumps:
        errors.append(
            f"missing {spec['dump_subdir']}/{final_step}.jsonl under {spec['log_roots']} "
            "(the run must reach its final evaluation with trainer.validation_data_dir as shipped)"
        )
    else:
        source = dumps[0]
        rows, correct, problems = score_dump(source)
        if rows < spec["min_rows"]:
            errors.append(f"{source.relative_to(results)} has {rows} scored rows, fewer than the {spec['min_rows']} required (full GSM8K test split)")
        else:
            judged = Decimal(correct) / Decimal(rows)
            if judged < required:
                errors.append(f"boxed held-out accuracy {judged:.4f} at step {final_step} is below required {required}")
            elif required_max is not None and judged > required_max:
                errors.append(f"boxed held-out accuracy {judged:.4f} at step {final_step} is above the maximum {required_max}")
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
        "dump_problems": problems[:20],
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        metrics = {"score": 0.0, "case_score": 0.0, "passed": False, "errors": [f"verifier error: {exc}"], "deterministic": True, "judge_used": False}
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
