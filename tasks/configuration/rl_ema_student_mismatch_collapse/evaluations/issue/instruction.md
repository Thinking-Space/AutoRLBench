# Issue 诊断指令（给被测 agent）

## 背景
一个 RL（GRPO 类）训练在 VLA/LIBERO-Long 上，三组 credit 变体（B1/A-gae/A-uniform）全在 SFT(44%) 以下，且 A-gae 未超 A-uniform。同目录提供：student 权重评测、EMA 权重评测、B1 训练曲线、SFT 基线、训练配置。

## 你的任务
1. 阅读 `../training_report.txt` 与 `../metrics_summary.json`。
2. 用 `tools/inspect_logs.py` 检查评测结果与训练曲线，阅读 `configs/`。
3. **定位三组 RL 全低于 SFT 的根因**，并给出支撑证据。
4. 输出诊断报告。

## 规则
- 只依据本 workspace 提供的材料。
- 注意区分"方法不行"与"评测/训练配置问题"。