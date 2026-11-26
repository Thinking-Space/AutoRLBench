#!/usr/bin/env python3
"""oracle/apply_fix.py —— 修复参考：评 EMA + 修 weight scale，展示恢复。"""
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
    student = load(os.path.join(LOGS, "eval_student.jsonl"))
    ema = load(os.path.join(LOGS, "eval_ema.jsonl"))
    curve = load(os.path.join(LOGS, "train_curve_b1.jsonl"))
    print("修复前(student 评测):", {r["arm"]: r["success_rate"] for r in student})
    print("修复目标(EMA 评测):", {r["arm"]: r["success_rate"] for r in ema})
    print(f"B1 训练曲线: step5={curve[4]['success_rate']}, step100={curve[-1]['success_rate']} (崩溃自锁)")
    print("修复 = 评 EMA 权重 + 修 nft_weight_scale(能量对称) → RL 臂≥SFT、训练不崩。")
    print("Rerun 阈值：至少一个 arm(EMA)≥SFT，B1 曲线不归零。")


if __name__ == "__main__":
    main()