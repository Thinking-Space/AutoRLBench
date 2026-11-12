#!/usr/bin/env python3
"""Summarize harbor-evaluate runs and apply the validity rules.

Walks <output-root>/<task_id>/<agent>-<model>/<timestamp>/report.json (the
default root is outputs/harbor-evaluate), keeps the newest run per
(task, agent-model, level), and prints one line per trial plus totals.

  python3 skills/harbor-evaluate/examples/collect-results.py \
      [--output-root DIR] [--model MODEL] [--level issue|solution]

Validity (a run is counted only when valid):
  issue     -> metrics.status == "judged" and no infrastructure_errors
               (status "unavailable" means the judge call failed: re-run the
               judge locally or re-run the trial, do not count it as 0)
  solution  -> no infrastructure_errors (score 0 with metrics.failure such as
               "agent did not submit report.md" or "system rerun failed" is a
               real zero, not an invalid run)

Exit code: 0 when every collected run is valid, 1 when any run is invalid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def trial_row(trial: dict) -> dict | None:
    """Normalize one trial dict into a display row, or None if unreadable."""
    metrics = trial.get("metrics") or {}
    level = trial.get("evaluation_level")
    infra = metrics.get("infrastructure_errors")
    if level == "issue":
        valid = metrics.get("status") == "judged" and not infra
    else:
        valid = not infra

    note = ""
    if infra:
        note = "infra: " + "; ".join(str(e) for e in infra)[:80]
    elif level == "issue" and metrics.get("status") != "judged":
        note = "judge {}: {}".format(metrics.get("status"), str(metrics.get("error", ""))[:80])
    elif metrics.get("failure"):
        note = str(metrics["failure"])

    score = trial.get("score")
    row = {
        "task": trial.get("task_id", "?"),
        "level": level,
        "passed": trial.get("passed"),
        "score": score,
        "valid": valid,
        "note": note,
    }
    for key in ("verdict_score", "root_cause_score", "evidence_score"):
        if key in metrics:
            row[key] = metrics[key]
    return row


def collect(root: Path, model_filter: str | None, level_filter: str | None) -> list[dict]:
    """Newest report per (task, label, level) under root."""
    best: dict[tuple, tuple] = {}
    # The timestamp component of the run path (<output_root>/<task_id>/<label>/<YYYYmmdd-HHMMSS>/)
    # is lexicographically ordered, so it is a more reliable recency signal than report mtime.
    for report_path in sorted(root.glob("*/*/*/report.json")):
        task_id = report_path.parts[-4]
        label = report_path.parts[-3]
        stamp = report_path.parts[-2]
        if model_filter and model_filter not in label:
            continue
        try:
            data = json.loads(report_path.read_text())
        except (OSError, ValueError):
            continue
        for trial in data.get("trials", []):
            level = trial.get("evaluation_level")
            if level_filter and level != level_filter:
                continue
            key = (task_id, label, level)
            if key not in best or stamp > best[key][0]:
                row = trial_row(trial)
                if row is not None:
                    best[key] = (stamp, row)
    return [row for _, row in best.values()]


def fmt(value, width=7, ndigits=3) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, bool):
        return ("PASS" if value else "FAIL").rjust(width)
    if isinstance(value, (int, float)):
        return f"{value:.{ndigits}f}".rjust(width)
    return str(value)[:width].rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="outputs/harbor-evaluate", type=Path)
    parser.add_argument("--model", default=None, help="filter on the <agent>-<model> directory label (substring)")
    parser.add_argument("--level", default=None, choices=["issue", "solution"])
    args = parser.parse_args()

    if not args.output_root.is_dir():
        print(f"no runs under {args.output_root}", file=sys.stderr)
        return 2
    rows = collect(args.output_root, args.model, args.level)
    if not rows:
        print("no collected runs (check --model / --level filters)", file=sys.stderr)
        return 2

    has_subscores = any("verdict_score" in r for r in rows)
    header = f"{'task':44s} {'lvl':8s} {'pass':4s} {'score':6s}"
    if has_subscores:
        header += f" {'verdict':7s} {'rc':7s} {'ev':7s}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["level"] or "", r["task"])):
        line = (
            f"{row['task']:44s} {str(row['level']):8s} "
            f"{'PASS' if row['passed'] else 'FAIL':4s} {fmt(row['score'])}"
        )
        if has_subscores:
            line += (
                f" {fmt(row.get('verdict_score'))}"
                f" {fmt(row.get('root_cause_score'))}"
                f" {fmt(row.get('evidence_score'))}"
            )
        if not row["valid"]:
            line += "  [INVALID]"
        print(line)
        if row["note"]:
            print(f"{'':44s} {row['note']}")

    print("-" * len(header))
    for level in ("issue", "solution"):
        subset = [r for r in rows if r["level"] == level]
        if not subset:
            continue
        valid = [r for r in subset if r["valid"]]
        passed = sum(1 for r in valid if r["passed"])
        rate = 100.0 * passed / len(valid) if valid else 0.0
        print(
            f"{level:8s}: {passed}/{len(valid)} valid runs pass ({rate:.1f}%), "
            f"{len(subset) - len(valid)} invalid run(s) -- rerun those"
        )

    return 1 if any(not r["valid"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
