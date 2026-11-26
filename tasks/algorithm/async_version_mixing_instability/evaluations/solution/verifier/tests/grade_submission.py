#!/usr/bin/env python3
# evaluations/solution/verifier/tests/grade_submission.py —— 素材阶段占位判分。
# 检查 agent 的修复产物是否满足：组内 version spread 归 0 + final reward 回升。
# 正式 task 时按 reference.md 替换为完整 grader。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_metrics.json"
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    ok_spread = m.get("group_version_spread_after_fix", 1.0) < 0.1
    ok_reward = m.get("final_reward_after_fix", 0.0) >= m.get("final_reward_threshold", 0.6)
    passed = ok_spread and ok_reward
    print(json.dumps({"passed": passed, "spread_ok": ok_spread, "reward_ok": ok_reward}))
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
