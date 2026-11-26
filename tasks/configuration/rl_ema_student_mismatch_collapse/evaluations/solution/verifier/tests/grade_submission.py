#!/usr/bin/env python3
# grade_submission.py —— 素材阶段占位判分。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_metrics.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    ok_arm = m.get("best_arm_ema_success", 0.0) >= m.get("sft_threshold", 0.44)
    ok_curve = m.get("b1_final_success", 1.0) > 0.0
    passed = ok_arm and ok_curve
    print(json.dumps({"passed": passed, "arm_ok": ok_arm, "curve_ok": ok_curve}))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()