#!/usr/bin/env python3
# judge.py —— 素材阶段占位判分：verdict/root_cause/evidence 0.2/0.6/0.2，及格 0.8。
import json
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnosis.md"
    text = open(path, encoding="utf-8").read().lower()

    # 根因关键词：优化侧配置
    opt_hits = sum(1 for kw in ["learning rate", "lr", "掩码", "masking", "batch", "学习率"] if kw.lower() in text)
    # 证据关键词：配置对比 / loss 曲线
    ev_hits = sum(1 for kw in ["config", "配置", "loss", "震荡", "oscillat", "曲线"] if kw.lower() in text)
    # 是否识破混淆项
    trap = ("epoch" in text or "混淆" in text or "confound" in text or "对照" in text)
    # 错因：只归因数据质量
    data_only = ("data quality" in text or "数据质量" in text) and opt_hits == 0

    verdict = 1.0 if (opt_hits > 0 and not data_only) else 0.2
    root_cause = min(1.0, opt_hits / 3.0)
    evidence = min(1.0, ev_hits / 3.0)
    if trap:
        evidence = min(1.0, evidence + 0.2)
    if data_only:
        verdict *= 0.3
    score = 0.2 * verdict + 0.6 * root_cause + 0.2 * evidence
    print(json.dumps({"score": round(score, 3), "verdict": round(verdict, 2),
                      "root_cause": round(root_cause, 2), "evidence": round(evidence, 2)}))
    sys.exit(0 if score >= 0.8 else 1)

if __name__ == "__main__":
    main()