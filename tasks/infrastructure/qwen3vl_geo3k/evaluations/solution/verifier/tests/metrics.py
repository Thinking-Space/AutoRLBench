"""Numerical gates. These functions never read source text to award correctness."""
import math


def diff_stats(a, b):
    if not a or len(a) != len(b):
        raise ValueError("empty or misaligned per-token arrays")
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in [*a, *b]):
        raise ValueError("non-finite per-token values")
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    worst = max(range(len(diffs)), key=diffs.__getitem__)
    return {"max_abs_diff": diffs[worst], "mean_abs_diff": sum(diffs) / len(diffs), "worst_token_index": worst}


def validate_records(data, spec):
    records = data["records"]
    names = [row["name"] for row in records]
    if len(names) != len(set(names)) or set(names) != set(spec["cases"]):
        raise ValueError("missing, duplicated, or unexpected cases")
    results = []
    for row in records:
        if len(row["response_ids"]) != spec["response_tokens"]:
            raise ValueError("wrong response token count")
        if len(row["trainer_logps"]) != len(row["response_ids"]):
            raise ValueError("trainer output token count differs")
        if row["engine_vs_reference_positions_max_diff"] != 0:
            positions_ok = False
        else:
            positions_ok = True
        results.append({"name": row["name"], "positions_correct": positions_ok,
                        **diff_stats(row["vllm_logps"], row["trainer_logps"])})
    return results


def repair_verdict(data, spec):
    rows = validate_records(data, spec)
    passed = all(row["positions_correct"] and row["max_abs_diff"] < spec["baseline_max_nat"] for row in rows)
    return {"passed": passed, "score": float(passed), "records": rows,
            "metric": spec["metric"], "baseline_limit_nat": spec["baseline_max_nat"]}


def paired_verdict(candidate, healthy, spec):
    if candidate["capture_sha256"] != healthy["capture_sha256"]:
        raise ValueError("paired runs did not score the same immutable rollout capture")
    bad = validate_records(candidate, spec)
    good = validate_records(healthy, spec)
    control = next(row for row in bad if row["name"] == "text_control")
    visual = [row for row in bad if row["name"] != "text_control"]
    worst_bad = max(row["max_abs_diff"] for row in visual)
    worst_good = max(row["max_abs_diff"] for row in good)
    ratio = worst_bad / max(worst_good, 1e-8)
    conditions = {
        "healthy_baseline_passes": repair_verdict(healthy, spec)["passed"],
        "candidate_text_control_passes": control["max_abs_diff"] < spec["baseline_max_nat"] and control["positions_correct"],
        "candidate_has_wrong_image_positions": all(not row["positions_correct"] for row in visual),
        "candidate_has_severe_mismatch": worst_bad >= spec["severe_max_nat"],
        "paired_separation_passes": ratio >= spec["min_separation_ratio"],
    }
    return {"qualified": all(conditions.values()), "conditions": conditions,
            "candidate_image_max_nat": worst_bad, "healthy_max_nat": worst_good, "separation_ratio": ratio,
            "candidate": bad, "healthy": good}
