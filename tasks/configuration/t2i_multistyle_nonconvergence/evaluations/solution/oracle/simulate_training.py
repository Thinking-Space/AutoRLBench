"""
simulate_training.py  (SOLUTION side — 评测方使用，不要放进 agent 的 workspace)

生成一个"多风格 T2I SFT 训练不收敛"的、自洽的可诊断 run。
背景（真实排查经验）：
  - MMU / 单风格 / 小数据(4/8/16风格) 都能收敛；
  - 全量数据(~14.5万张) 收敛过慢、loss 震荡。
根因（两个正交杠杆叠加）：
  优化侧：LR 1.5e-6 过低 + 掩码率 0.05~0.25 过窄 + 有效 batch 偏小 → 梯度方差大、多峰数据喂不动；
  数据侧：全量混入弱风格/错标/重复/caption 不匹配 → 有效信号占比低。
诊断陷阱：小数据能收敛，一部分是因为它被反复训练了几十上百 epoch（等效训得更充分），
  所以"小数据好"不能直接等同于"数据质量是唯一主因"——需要对照实验区分。

产物（写到 workspace/）：
  frozen_evidence/loss_full_defective.jsonl   全量 + 坏配置（待诊断）
  frozen_evidence/loss_full_reference.jsonl   全量 + 好配置（对照）
  frozen_evidence/loss_small_defective.jsonl  小数据 + 坏配置（混淆项：能收敛）
  configs/*.yaml                               各 run 的训练配置
  training_report.txt, metrics_summary.json    给 agent 的中性输入
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
WS = os.path.join(TASK, "workspace")
LOGS = os.path.join(WS, "frozen_evidence")
CONFIGS = os.path.join(WS, "configs")


def simulate_loss(n_steps, lr, mask_lo, mask_hi, batch, n_styles, seed, epochs_equiv=1.0):
    """简化 loss 模型：
       loss(t) = irreducible + A*exp(-rate*t) + oscillation + noise
       rate 随 LR 增大而增大；oscillation 随 batch 减小、掩码变窄而增大；
       irreducible 随掩码变窄、风格数增多而升高。
       epochs_equiv: 等效训练轮数（小数据被反复训练 → 等效更多步 → 收敛）。"""
    rng = np.random.default_rng(seed)
    # 有效步数（小数据等效更多步）
    eff_steps = int(n_steps * epochs_equiv)
    # 收敛速率：LR 越高越快
    rate = 0.02 * (lr / 1.5e-6) ** 0.5
    # 不可约损失：掩码越窄、风格越多越高
    mask_width = mask_hi - mask_lo
    irreducible = 0.15 + 0.5 * (1.0 - min(mask_width / 0.85, 1.0)) + 0.02 * n_styles
    # 初始幅度
    A = 1.2
    # 震荡幅度：batch 越小越大
    osc_amp = 0.15 * (32.0 / max(batch, 1)) ** 0.5
    losses = []
    for t in range(1, n_steps + 1):
        base = irreducible + A * np.exp(-rate * min(t, eff_steps))
        osc = osc_amp * np.sin(t * 0.8) * np.exp(-t / 200.0)
        noise = rng.normal(0, 0.02)
        losses.append(float(max(base + osc + noise, 0.01)))
    return losses


def write_jsonl(path, losses):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in enumerate(losses, 1):
            f.write(json.dumps({"step": t, "loss": round(l, 5)}) + "\n")


def write_config(path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in kw.items():
            f.write(f"{k}: {v}\n")


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(CONFIGS, exist_ok=True)
    n = 200

    # 待诊断：全量 + 坏配置（LR 低、掩码窄、batch 小）
    defective = simulate_loss(n, lr=1.5e-6, mask_lo=0.05, mask_hi=0.25, batch=16, n_styles=16, seed=1)
    # 对照：全量 + 好配置（LR 高、掩码宽、batch 大）
    reference = simulate_loss(n, lr=1.0e-5, mask_lo=0.05, mask_hi=0.90, batch=48, n_styles=16, seed=2)
    # 混淆项：小数据 + 坏配置，但等效更多 epoch → 能收敛
    small = simulate_loss(n, lr=1.5e-6, mask_lo=0.05, mask_hi=0.25, batch=16, n_styles=4, seed=3, epochs_equiv=6.0)

    write_jsonl(os.path.join(LOGS, "loss_full_defective.jsonl"), defective)
    write_jsonl(os.path.join(LOGS, "loss_full_reference.jsonl"), reference)
    write_jsonl(os.path.join(LOGS, "loss_small_defective.jsonl"), small)

    write_config(os.path.join(CONFIGS, "full_defective.yaml"),
                 learning_rate="1.5e-6", masking_rate_min=0.05, masking_rate_max=0.25,
                 grad_accum=16, max_grad_norm=0.5, n_styles=16, dataset="full_145k")
    write_config(os.path.join(CONFIGS, "full_reference.yaml"),
                 learning_rate="1.0e-5", masking_rate_min=0.05, masking_rate_max=0.90,
                 grad_accum=48, max_grad_norm=1.0, n_styles=16, dataset="full_145k")
    write_config(os.path.join(CONFIGS, "small_defective.yaml"),
                 learning_rate="1.5e-6", masking_rate_min=0.05, masking_rate_max=0.25,
                 grad_accum=16, max_grad_norm=0.5, n_styles=4, dataset="small_4styles",
                 note="trained many more epochs")

    # 给 agent 的中性报告
    def tail(xs, k=20):
        return float(np.mean(xs[-k:]))

    report = f"""# 训练报告（待诊断）

## 设置
- 任务：多风格 T2I 风格化 SFT（16 风格，全量约 14.5 万张）
- 现象：loss 不收敛 / 收敛过慢、震荡

## 我观察到的
- 全量数据 + 当前配置：loss 末尾均值 {tail(defective):.3f}，震荡明显，下降缓慢
- 全量数据 + 另一套配置（对照）：loss 末尾均值 {tail(reference):.3f}，下降快且低
- 小数据（4 风格）+ 当前配置：loss 能收敛（末尾均值 {tail(small):.3f}）
- MMU（理解任务）、T2I 单风格：都能收敛

## 观察
小数据（4 风格）+ 当前配置能收敛；MMU、单风格也能收敛。

## 问题
全量多风格 T2I 为什么学不动 / 震荡？请定位根因，并给出支撑证据。
配置在 configs/ 下，loss 曲线在 frozen_evidence/ 下。
"""
    with open(os.path.join(WS, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "run_under_diagnosis": "full_defective",
        "symptom": "全量多风格 T2I SFT loss 不收敛/震荡，收敛过慢",
        "agent_visible_metrics": {
            "final_loss_full_defective": round(tail(defective), 4),
            "final_loss_full_reference": round(tail(reference), 4),
            "final_loss_small_defective": round(tail(small), 4),
            "note": "小数据(4风格)+当前配置能收敛；MMU、单风格也能收敛",
        },
        "available_raw_materials": [
            "frozen_evidence/loss_full_defective.jsonl",
            "frozen_evidence/loss_full_reference.jsonl",
            "frozen_evidence/loss_small_defective.jsonl",
            "configs/full_defective.yaml, full_reference.yaml, small_defective.yaml",
        ],
    }
    with open(os.path.join(WS, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== 自洽性检查 ===")
    print(f"final loss  defective={tail(defective):.3f}  reference={tail(reference):.3f}  "
          f"-> defective 更高(更差): {tail(defective) > tail(reference)}")
    print(f"small-data(坏配置) final={tail(small):.3f}  -> 能收敛(低于 defective): {tail(small) < tail(defective)}")
    print(f"震荡幅度 defective 末段 std={np.std(defective[-40:]):.3f}  "
          f"reference 末段 std={np.std(reference[-40:]):.3f}")


if __name__ == "__main__":
    main()