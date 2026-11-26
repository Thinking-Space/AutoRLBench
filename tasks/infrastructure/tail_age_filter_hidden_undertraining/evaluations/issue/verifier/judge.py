#!/usr/bin/env python3
# judge.py —— 素材阶段占位判分：verdict/root_cause/evidence 0.2/0.6/0.2，及格 0.8。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnosis.md"
    text = open(path, encoding="utf-8").read().lower()

    rc_hits = sum(1 for kw in ["age", "门控", "filter", "过滤", "wall-time", "wall time", "长任务", "long task"] if kw.lower() in text)
    ev_hits = sum(1 for kw in ["horizon", "分层", "stratif", "长程", "long-horizon", "按 horizon"] if kw.lower() in text)
    # 是否识破聚合陷阱
    trap = ("聚合" in text or "aggregate" in text) and ("看不出" in text or "掩盖" in text or "mask" in text or "hidden" in text)
    # 错因：只看聚合说没问题
    wrong = ("没问题" in text or "no problem" in text or "正常" in text) and ev_hits == 0

    verdict = 1.0 if (rc_hits > 0 and not wrong) else 0.2
    root_cause = min(1.0, rc_hits / 3.0)
    evidence = min(1.0, ev_hits / 3.0)
    if trap:
        evidence = min(1.0, evidence + 0.2)
    if wrong:
        verdict *= 0.3
    score = 0.2 * verdict + 0.6 * root_cause + 0.2 * evidence
    print(json.dumps({"score": round(score, 3), "verdict": round(verdict, 2),
                      "root_cause": round(root_cause, 2), "evidence": round(evidence, 2)}))
    sys.exit(0 if score >= 0.8 else 1)

if __name__ == "__main__":
    main()