#!/usr/bin/env python3
# grade_submission.py —— 素材阶段占位判分：检查修复后末段 loss 与震荡是否达标。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_metrics.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    ok_loss = m.get("final_loss_after_fix", 1.0) <= m.get("final_loss_threshold", 0.55)
    ok_osc = m.get("oscillation_std_after_fix", 1.0) <= m.get("oscillation_threshold", 0.05)
    passed = ok_loss and ok_osc
    print(json.dumps({"passed": passed, "loss_ok": ok_loss, "osc_ok": ok_osc}))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()