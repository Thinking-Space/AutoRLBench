#!/usr/bin/env python3
# judge.py —— 素材阶段占位判分：verdict/root_cause/evidence 0.2/0.6/0.2，及格 0.8。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnosis.md"
    text = open(path, encoding="utf-8").read().lower()

    # 根因关键词：EMA/student + 训练崩溃
    ema_hits = sum(1 for kw in ["ema", "student", "权重", "weights", "checkpoint"] if kw.lower() in text)
    collapse_hits = sum(1 for kw in ["崩溃", "collapse", "nft_weight_scale", "能量对称", "energy", "自锁", "归零"] if kw.lower() in text)
    ev_hits = sum(1 for kw in ["eval", "评测", "曲线", "curve", "sft", "baseline"] if kw.lower() in text)
    # 错因：只归因 credit 方法
    wrong = ("credit" in text or "方法不行" in text) and ema_hits == 0 and collapse_hits == 0

    verdict = 1.0 if ((ema_hits > 0 or collapse_hits > 0) and not wrong) else 0.2
    root_cause = min(1.0, (ema_hits + collapse_hits) / 4.0)
    evidence = min(1.0, ev_hits / 3.0)
    if wrong:
        verdict *= 0.3
    score = 0.2 * verdict + 0.6 * root_cause + 0.2 * evidence
    print(json.dumps({"score": round(score, 3), "verdict": round(verdict, 2),
                      "root_cause": round(root_cause, 2), "evidence": round(evidence, 2)}))
    sys.exit(0 if score >= 0.8 else 1)

if __name__ == "__main__":
    main()