#!/usr/bin/env python3
"""
oracle/apply_fix.py —— 修复参考实现（配置修正 + 预期效果演示）。
读取 defective 与 reference 的 loss，展示修复后配置应达到的水平。
"""
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
                rows.append(json.loads(line)["loss"])
    return rows


def main():
    defective = load(os.path.join(LOGS, "loss_full_defective.jsonl"))
    reference = load(os.path.join(LOGS, "loss_full_reference.jsonl"))
    print(f"修复前 defective 末段 loss={np.mean(defective[-20:]):.3f}  std={np.std(defective[-40:]):.3f}")
    print(f"修复目标 reference 末段 loss={np.mean(reference[-20:]):.3f}  std={np.std(reference[-40:]):.3f}")
    print("修复 = LR↑ + 掩码放宽 + batch↑ + grad clip↑ + shuffle → 末段 loss 降低、震荡收敛。")
    print("Rerun 阈值：末段 loss 接近 reference，std 显著下降。")


if __name__ == "__main__":
    main()