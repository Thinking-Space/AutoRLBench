"""
simulate_training.py  (SOLUTION side — 评测方使用，不要放进 agent 的 workspace)

生成"RL 三臂全低于 SFT，但根因不是 credit 方法，而是 EMA/student 评错权重 + 训练崩溃"的
自洽可诊断 run。基于真实排查经验（M3 重跑清单）。

机制：
  - 训练用 EMA 采样，但上一轮评的是 student 权重 → 评错权重，RL 臂看起来全低于 SFT；
  - 训练本身也崩了：nft_weight_scale 把 mse 形式 [1.0, 0.2] 抄进 dpo 形式 → 几步内成功率归零、自锁；
  - 诊断陷阱：三组 RL 全低于 SFT(44%)，容易误判"credit 方法不行"，实际是评错权重+训练崩。

产物（workspace/）：
  frozen_evidence/eval_student.jsonl   评 student 权重（待诊断，全低于 SFT）
  frozen_evidence/eval_ema.jsonl       评 EMA 权重（对照，≥SFT）
  frozen_evidence/sft_baseline.json    SFT 基线
  frozen_evidence/train_curve_b1.jsonl B1 臂训练曲线（崩溃可见）
  configs/*.yaml
  training_report.txt, metrics_summary.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WS = os.path.join(TASK, "workspace")
LOGS = os.path.join(WS, "frozen_evidence")
CONFIGS = os.path.join(WS, "configs")

SFT_BASELINE = 0.44


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(CONFIGS, exist_ok=True)
    rng = np.random.default_rng(0)

    # 评 student 权重：三臂全低于 SFT（待诊断）
    arms = ["B1", "A-gae", "A-uniform"]
    student_scores = {"B1": 0.22, "A-gae": 0.344, "A-uniform": 0.374}
    ema_scores = {"B1": 0.46, "A-gae": 0.48, "A-uniform": 0.47}  # EMA ≥ SFT

    write_jsonl(os.path.join(LOGS, "eval_student.jsonl"),
                [{"arm": a, "success_rate": student_scores[a], "weights": "student"} for a in arms])
    write_jsonl(os.path.join(LOGS, "eval_ema.jsonl"),
                [{"arm": a, "success_rate": ema_scores[a], "weights": "ema"} for a in arms])
    with open(os.path.join(LOGS, "sft_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"sft_success_rate": SFT_BASELINE}, f, indent=2)

    # B1 训练曲线：几步内成功率归零（能量对称违反 → 自锁）
    curve = []
    sr = SFT_BASELINE
    for step in range(1, 101):
        if step <= 5:
            sr = max(0.0, sr - 0.08)   # 几步内崩到 0
        else:
            sr = 0.0                    # 自锁
        curve.append({"step": step, "success_rate": round(sr, 4), "reward": round(sr, 4)})
    write_jsonl(os.path.join(LOGS, "train_curve_b1.jsonl"), curve)

    write_config(os.path.join(CONFIGS, "b1.yaml"),
                 nft_weight_scale="[1.0, 0.2]", note="mse form copied into dpo form",
                 eval_weights="student", ema_decay=0.999)
    write_config(os.path.join(CONFIGS, "reference.yaml"),
                 nft_weight_scale="[1.0, 0.2]", eval_weights="ema", ema_decay=0.999,
                 note="correct: evaluate EMA weights")

    report = f"""# 训练报告（待诊断）

## 设置
- 算法：RL（GRPO 类）在 VLA/LIBERO-Long 上，三组 credit 变体：B1(终局)/A-gae(逐chunk GAE)/A-uniform(轨迹级)
- 基线：SFT 成功率 {SFT_BASELINE}
- 现象：三组 RL 全在 SFT 以下 —— B1 {student_scores['B1']}、A-gae {student_scores['A-gae']}、A-uniform {student_scores['A-uniform']}
  且 A-gae 没超 A-uniform

## 我的困惑
是不是 credit assignment 方法不行？还是别的原因？

## 可用材料
- frozen_evidence/eval_student.jsonl（当前评测结果）
- frozen_evidence/eval_ema.jsonl（另一套权重的评测）
- frozen_evidence/train_curve_b1.jsonl（B1 训练曲线）
- frozen_evidence/sft_baseline.json
- configs/（训练配置）

## 问题
三组 RL 为什么全低于 SFT？请定位根因，并给出支撑证据。
"""
    with open(os.path.join(WS, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "run_under_diagnosis": "student_eval",
        "symptom": "三组 RL 全低于 SFT(44%)，A-gae 未超 A-uniform",
        "agent_visible_metrics": {
            "sft_baseline": SFT_BASELINE,
            "student_scores": student_scores,
            "note": "训练用 EMA 采样；当前评的是 student 权重",
        },
        "available_raw_materials": [
            "frozen_evidence/eval_student.jsonl",
            "frozen_evidence/eval_ema.jsonl",
            "frozen_evidence/train_curve_b1.jsonl",
            "frozen_evidence/sft_baseline.json",
            "configs/b1.yaml, reference.yaml",
        ],
    }
    with open(os.path.join(WS, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== 自洽性检查 ===")
    print(f"student 评测: B1={student_scores['B1']} A-gae={student_scores['A-gae']} A-uniform={student_scores['A-uniform']} -> 全<SFT({SFT_BASELINE}): "
          f"{all(v < SFT_BASELINE for v in student_scores.values())}")
    print(f"EMA 评测: B1={ema_scores['B1']} A-gae={ema_scores['A-gae']} -> ≥SFT: {all(v >= SFT_BASELINE for v in ema_scores.values())}")
    print(f"B1 训练曲线: step5={curve[4]['success_rate']}, step100={curve[-1]['success_rate']} -> 崩溃自锁: {curve[-1]['success_rate']==0.0}")


def write_config(path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in kw.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()