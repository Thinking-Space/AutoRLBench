#!/usr/bin/env python3
"""oracle/apply_fix.py —— 修复参考：放宽 age 门控，展示长程成功率回升。"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGS = os.path.join(TASK_ROOT, "workspace", "frozen_evidence")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    tight = load(os.path.join(LOGS, "metrics_tight.jsonl"))
    relaxed = load(os.path.join(LOGS, "metrics_relaxed.jsonl"))
    long_tight = np.mean([r["success_rate"] for r in tight if r["horizon"] > 10])
    long_relaxed = np.mean([r["success_rate"] for r in relaxed if r["horizon"] > 10])
    print(f"修复前 长程(h>10)成功率 tight={long_tight:.3f}")
    print(f"修复目标(放宽门控) 长程成功率={long_relaxed:.3f}")
    print("修复 = 放宽 age_limit / coverage-aware 调度 → 长任务被准入 → 长程成功率回升。")
    print("Rerun 阈值：长程成功率接近 relaxed 水平，聚合不降。")


if __name__ == "__main__":
    main()