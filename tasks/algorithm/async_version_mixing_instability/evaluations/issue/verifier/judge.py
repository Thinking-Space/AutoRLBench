#!/usr/bin/env python3
# judge.py —— 素材阶段占位判分：verdict/root_cause/evidence 0.2/0.6/0.2，及格 0.8。
# 正式 task 时按 reference.md / verifier 规范替换为完整 LLM-judge 或规则判分。
import json
import re
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnosis.md"
    text = open(path, encoding="utf-8").read().lower()

    # 根因关键词
    rc_hits = sum(1 for kw in ["behavior_version", "policy version", "版本混合", "multiple versions",
                               "group", "baseline"] if kw.lower() in text)
    ev_hits = sum(1 for kw in ["spread", "version", "混合", "mixing", "组内"] if kw.lower() in text)
    wrong = sum(1 for kw in ["learning rate", "数据", "data quality", "reward hack"] if kw.lower() in text)

    verdict = 1.0 if ("group" in text or "版本" in text or "version" in text) else 0.2
    root_cause = min(1.0, rc_hits / 4.0)
    evidence = min(1.0, ev_hits / 4.0)
    if wrong:
        verdict *= 0.5
    score = 0.2 * verdict + 0.6 * root_cause + 0.2 * evidence
    print(json.dumps({"score": round(score, 3), "verdict": round(verdict, 2),
                      "root_cause": round(root_cause, 2), "evidence": round(evidence, 2)}))
    sys.exit(0 if score >= 0.8 else 1)

if __name__ == "__main__":
    main()
