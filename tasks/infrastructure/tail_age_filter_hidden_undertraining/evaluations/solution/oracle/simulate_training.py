"""
simulate_training.py  (SOLUTION side — 评测方使用，不要放进 agent 的 workspace)

生成"异步 RL age 过滤导致长程能力隐形欠训练"的自洽可诊断 run。
机制：轨迹越长 → wall-time 越长 → version-lag 越大 → 越容易被 age 准入门控过滤。
于是能贡献梯度的系统性偏向短任务；聚合成功率(短任务主导)看不出问题，
只有按 horizon 切开才看到长程段塌陷。

产物（workspace/）：
  frozen_evidence/metrics_tight.jsonl     紧 age 门控(age_limit=4) —— 待诊断
  frozen_evidence/metrics_relaxed.jsonl   放宽门控(age_limit=12) —— 对照
  frozen_evidence/horizon_dist.json       真实任务 horizon 分布
  training_report.txt, metrics_summary.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WS = os.path.join(TASK, "workspace")
LOGS = os.path.join(WS, "frozen_evidence")

H = 20
T_STEP = 30.0
TAU = 60.0
LR = 0.02
hs = np.arange(1, H + 1)
PH = 0.72 ** hs
PH = PH / PH.sum()


def admit_lag(rng, h):
    return int(np.ceil(h * T_STEP * rng.lognormal(0.0, 0.12) / TAU))


def run(age_limit, seed, budget=40000):
    rng = np.random.default_rng(seed)
    c = np.zeros(H + 1)
    for _ in range(budget):
        h = rng.choice(hs, p=PH)
        if admit_lag(rng, h) <= age_limit:
            for hp in range(1, h + 1):
                c[hp] += LR * (1 - c[hp])
    return c[1:]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(LOGS, exist_ok=True)
    tight = np.array([run(4, s) for s in range(6)]).mean(0)     # age_limit=4
    relaxed = np.array([run(12, s) for s in range(6)]).mean(0)  # age_limit=12

    # 按 horizon 记录成功率
    write_jsonl(os.path.join(LOGS, "metrics_tight.jsonl"),
                [{"horizon": int(h), "success_rate": round(float(tight[h - 1]), 4)} for h in hs])
    write_jsonl(os.path.join(LOGS, "metrics_relaxed.jsonl"),
                [{"horizon": int(h), "success_rate": round(float(relaxed[h - 1]), 4)} for h in hs])
    with open(os.path.join(LOGS, "horizon_dist.json"), "w", encoding="utf-8") as f:
        json.dump({str(h): round(float(PH[h - 1]), 4) for h in hs}, f, ensure_ascii=False, indent=2)

    agg_tight = float((tight * PH).sum())
    agg_relaxed = float((relaxed * PH).sum())
    short_tight = float(tight[:5].mean())
    long_tight = float(tight[10:].mean())
    long_relaxed = float(relaxed[10:].mean())

    report = f"""# 训练报告（待诊断）

## 设置
- 算法：异步 RL（GRPO 类），带 age 准入门控（rollout 落后超过 age_limit 个版本则丢弃）
- 任务：长程 agent 任务，horizon 1~20，真实分布短任务占多数
- 现象：训练看起来正常，聚合成功率 {agg_tight:.3f}，与放宽门控的 {agg_relaxed:.3f} 几乎一样

## 我检查过的
- 聚合成功率：紧门控 {agg_tight:.3f} vs 放宽 {agg_relaxed:.3f} —— 差异很小，看不出问题
- 短任务(h<=5)成功率：{short_tight:.3f} —— 正常

## 问题
这个异步 RL 训练有没有被 age 门控"隐形地"损害某些能力？如果有，是什么、怎么发现？
metrics 在 frozen_evidence/ 下（按 horizon 分），horizon 分布也在。
"""
    with open(os.path.join(WS, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "run_under_diagnosis": "tight_age_filter",
        "symptom": "聚合成功率正常，但可能有隐形的能力欠训练",
        "agent_visible_metrics": {
            "aggregate_success_tight": round(agg_tight, 4),
            "aggregate_success_relaxed": round(agg_relaxed, 4),
            "short_task_success_tight": round(short_tight, 4),
            "note": "聚合指标被短任务主导，长程段需按 horizon 切开才可见",
        },
        "available_raw_materials": [
            "frozen_evidence/metrics_tight.jsonl",
            "frozen_evidence/metrics_relaxed.jsonl",
            "frozen_evidence/horizon_dist.json",
        ],
    }
    with open(os.path.join(WS, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== 自洽性检查 ===")
    print(f"聚合成功率  tight={agg_tight:.3f}  relaxed={agg_relaxed:.3f}  -> 差异小(看不出): {abs(agg_tight-agg_relaxed)<0.1}")
    print(f"长程成功率(h>10)  tight={long_tight:.3f}  relaxed={long_relaxed:.3f}  -> 塌陷暴露: {long_relaxed-long_tight>0.5}")
    print(f"短任务成功率  tight={short_tight:.3f}  -> 正常")


if __name__ == "__main__":
    main()