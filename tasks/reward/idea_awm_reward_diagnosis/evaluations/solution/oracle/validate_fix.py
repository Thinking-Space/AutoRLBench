#!/usr/bin/env python3
"""Validate the oracle reference fix for the solution-level replay grader.

Three checks:

1. The seven corrected rules classify all 14 replay cases correctly (each
   audited false-positive episode rejected, each synthesized correct
   completion accepted).
2. The original defective rules, replayed over the same 14 cases, accept
   every negative (the defect) while still accepting every positive (the
   cases are within reach of a repair) -- i.e. exactly 7/14.
3. The corrected rules survive the JSONL round-trip the grader performs:
   spliced into a copy of gen_verifier.pure_code.jsonl and loaded back by
   (scenario, task_idx), they still classify 14/14.

Run from the task root:  python3 evaluations/solution/oracle/validate_fix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "verifier" / "resources" / "tools"))
import replay_fixtures as rf  # noqa: E402

CASES = json.loads(
    (HERE.parent / "verifier" / "resources" / "replay_cases.json").read_text(encoding="utf-8")
)["cases"]

CORRECTED = HERE / "corrected_rules"
RULE_FILES = {
    ("task_management_16", 8): "task_management_16_8.py",
    ("logistics_and_shipment_tracking_1", 0): "logistics_and_shipment_tracking_1_0.py",
    ("stock_trading_1", 2): "stock_trading_1_2.py",
    ("iot_device_management_3", 5): "iot_device_management_3_5.py",
    ("subscription_management_4", 2): "subscription_management_4_2.py",
    ("fundraising_donations_5", 9): "fundraising_donations_5_9.py",
    ("discussion_forum_2", 0): "discussion_forum_2_0.py",
}

# The packaged (pre-fetch) copy of the defective verifier file, kept outside
# the task tree at design time by the task author; regenerate via the
# manifest pins if absent.
DEFECTIVE_JSONL = Path(
    __import__("os").environ.get(
        "AWM_DEFECTIVE_VERIFIER",
        str(Path(__file__).resolve().parents[6] / "outputs/idea-awm-design/gen_verifier.pure_code.jsonl"),
    )
)


def corrected_code(scenario, task_idx):
    return (CORRECTED / RULE_FILES[(scenario, task_idx)]).read_text(encoding="utf-8"), (
        f"<corrected {scenario}:{task_idx}>"
    )


def jsonl_code_factory(rules):
    def get(scenario, task_idx):
        entry = rf.find_rule(rules, scenario, task_idx)
        if entry is None:
            raise KeyError(f"no rule for {scenario}:{task_idx}")
        code = entry.get("verification", {}).get("code")
        if not code:
            raise KeyError(f"rule {scenario}:{task_idx} has no verification code")
        return code, f"<jsonl {scenario}:{task_idx}>"

    return get


def report(title, results, errors):
    correct = sum(1 for r in results.values() if r["correct"])
    total = len(results)
    print(f"== {title}: {correct}/{total} correct")
    for case_id, r in results.items():
        flag = "ok " if r["correct"] else "MISS"
        print(f"  [{flag}] {case_id:24s} verdict={r['verdict']:10s} expected={r['expected']}")
        if not r["correct"]:
            print(f"         detail: {r['detail'][:300]}")
    for error in errors:
        print(f"  [load] {error}")
    return correct, total


def main():
    failures = []

    # 1. corrected rules over all cases
    results, errors = rf.replay_all(corrected_code, CASES)
    correct, total = report("corrected rules", results, errors)
    if correct != total or errors:
        failures.append(f"corrected rules: {correct}/{total} (errors: {errors})")

    # 2. original defective rules over all cases
    defective_rules = rf.load_rules(DEFECTIVE_JSONL)
    results, errors = rf.replay_all(jsonl_code_factory(defective_rules), CASES)
    correct, total = report("original defective rules", results, errors)
    wrongly_accepted = [
        cid for cid, r in results.items() if r["expected"] == "incomplete" and r["verdict"] == "complete"
    ]
    lost_positives = [
        cid for cid, r in results.items() if r["expected"] == "complete" and r["verdict"] != "complete"
    ]
    if len(wrongly_accepted) != 7:
        failures.append(f"defective rules accepted {len(wrongly_accepted)} negatives (expected 7)")
    if lost_positives:
        failures.append(f"defective rules lost positives: {lost_positives}")

    # 3. JSONL round-trip: splice corrected rules into a copy and reload
    spliced = []
    replaced = set()
    for _, entry in defective_rules:
        entry = json.loads(json.dumps(entry))
        key = (entry.get("scenario"), entry.get("task_idx"))
        if key in RULE_FILES:
            entry["verification"]["code"] = corrected_code(*key)[0]
            replaced.add(key)
        spliced.append(entry)
    if replaced != set(RULE_FILES):
        failures.append(f"splice replaced {sorted(replaced)} != {sorted(RULE_FILES)}")
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        for entry in spliced:
            tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        spliced_path = tmp.name
    spliced_rules = rf.load_rules(spliced_path)
    results, errors = rf.replay_all(jsonl_code_factory(spliced_rules), CASES)
    correct, total = report("corrected rules via jsonl round-trip", results, errors)
    if correct != total or errors:
        failures.append(f"jsonl round-trip: {correct}/{total} (errors: {errors})")
    Path(spliced_path).unlink()

    if failures:
        print("\nVALIDATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nVALIDATION PASSED: 14/14 corrected, 7/14 defective (all negatives accepted), round-trip intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
