#!/usr/bin/env python3
# grade_submission.py —— 素材阶段占位判分：检查修复后长程成功率是否回升。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_metrics.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    ok_long = m.get("long_horizon_success_after_fix", 0.0) >= m.get("long_horizon_threshold", 0.8)
    ok_agg = m.get("aggregate_success_after_fix", 1.0) >= m.get("aggregate_threshold", 0.9)
    passed = ok_long and ok_agg
    print(json.dumps({"passed": passed, "long_ok": ok_long, "agg_ok": ok_agg}))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()