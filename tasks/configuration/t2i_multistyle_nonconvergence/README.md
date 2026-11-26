# AutoRLBench Task: t2i_multistyle_nonconvergence

- **Category**: `configuration`
- **Level**: issue（可升级 solution）
- **Domain**: 多风格 T2I 风格化 SFT（16 风格，全量约 14.5 万张）
- **One-liner**: 全量多风格 T2I SFT loss 不收敛/震荡；小数据能收敛是混淆项（等效更多 epoch），
  真正根因是优化侧配置（LR 过低 + 掩码率过窄 + 有效 batch 偏小）叠加数据侧噪声。

## 目录结构
```
t2i_multistyle_nonconvergence/
├── README.md
├── task.toml
├── workspace/                      ← 交给被测 agent（不含答案）
│   ├── training_report.txt
│   ├── metrics_summary.json
│   ├── frozen_evidence/            loss 曲线（defective / reference / small）
│   ├── configs/                    三套训练配置
│   └── tools/inspect_logs.py       中性查看器
├── environment/prepare.sh
└── evaluations/
    ├── issue/  instruction.md + oracle.md + verifier/
    └── solution/ instruction.md + oracle/ + verifier/
```

## 诊断陷阱
"小数据能收敛"会诱导 agent 归因到数据质量；但小数据收敛部分是因为被反复训练了更多 epoch
（等效训得更充分）。正确做法是用对照实验区分：全量数据上只改 LR，看 loss 是否下得更快更低。

## 根因（oracle 摘要）
优化侧：LR 1.5e-6 过低 + 掩码率 0.05~0.25 过窄 + 有效 batch 16 偏小 → 梯度方差大、多峰数据喂不动；
数据侧：全量混入弱风格/错标/重复/caption 不匹配 → 有效信号占比低。二者正交、可同时成立。

## 构造
`evaluations/solution/oracle/simulate_training.py` 生成 frozen_evidence 与 configs（模拟 run，自洽可复现）。