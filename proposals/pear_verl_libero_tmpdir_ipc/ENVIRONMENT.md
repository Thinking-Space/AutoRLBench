# Environment and reproducibility / 环境与可复现性

## 已知实验条件

| 项目 | 已知记录 |
| --- | --- |
| 模型 | Qwen/Qwen3.8-27B |
| 模型revision | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 |
| PEAR commit | f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087，另有本任务overlay |
| Harness commit | 1c31cbbbf54cadbb53db64e381ac69d77e74e8ea，另有本任务适配 |
| veRL | 基础8e03c039f9a70750490c93196b759b332fc029fd，官方补丁HEAD e5632356b73272aaf4afcd78bbe04aae240599fd，另有适配overlay |
| LIBERO commit | 8f1084e3132a39270c3a13ebe37270a43ece2a01 |
| 资源 | 8节点×8H20；2训练节点、6rollout节点 |
| 训练/采样 | FSDP2、SP8、separate_async；12个TP4推理副本 |
| 实际batch | 16group×8轨迹，128非padding；不是原配方128×8 |
| 恢复来源 | 已验收CP15；初始policy15随后更新到16/17 |
| 优化器/采样 | lr1e-6；训练temperature1/top_p1；top32 |
| 预算 | prompt6144、response160000 |
| harness工具 | move、hold、capture_scene、finish；两路512RGB，深度关闭 |
| 指标保存 | 原始JSONL、TensorBoard、离线W&B；本包只交第16/17步原始JSONL摘录 |
| 保留策略 | save1/keep1；本包没有复制训练checkpoint |

版本来自工作区已有的实验固定条件记录；本次没有重新checkout或获取模型。它们与本地overlay共同构成运行状态，**不能把上游commit表当成完整源码锁定**。本包精确锁定并提供的本地变更只有两份调度模块的修复前后快照。

故障与恢复的直接launch配置摘录分别见failure_launch.json、recovery_launch.json。其中协调端TMPDIR在两侧仍可能相同：主修复改变的是生成Ray远端runtime_env的转发列表。完整源码和主补丁能显示这一点。

## 指定镜像适配状态

当前运行来自原有私有集群环境。**尚无证据证明故障侧在AutoRLBench八个指定镜像之一中执行。** 不填写虚构的image digest，不把已有日志改名为指定镜像复现。

references/ENVIRONMENTS.md保留当日官方列表；应选择最接近的栈并验证额外依赖、仿真资产、Ray后端和调度契约。没有新增自定义镜像例外。

本地共享虚拟环境解释器在开发机路径下无法直接执行，因此本次没有用它读取版本或运行生产检查；新增探针仅依赖系统Python标准库，未安装包。这里不猜测历史torch/vLLM/CUDA等精确版本。

## 完成正式入库前仍需补齐

| 事项 | 当前状态 | 下一步 |
| --- | --- | --- |
| 指定镜像中的故障复现 | 未完成 | 在选定共享镜像运行同根因完整实验并保留直接日志 |
| 完整可运行RL workspace | 未打包 | 汇入明确版本的框架、全部必要overlay、入口和仿真/数据契约 |
| CP15来源可复现 | 内部历史恢复点，有证据；载荷未交付 | 明确是否重建前15步、使用许可允许的固定资产，或证明可在更小同根因实验中复现 |
| 资产准备 | 已列已知版本，未生成可独立执行prepare.sh | 补齐模型/数据/仿真/依赖的固定获取和重建方式 |
| 原始完整日志与W&B | 本包为直接traceback、CPU日志、选取元数据和两步原始指标 | 正式raw材料按需求补齐完整历史运行证据；不将摘录称为完整W&B |
| startup rerun门槛 | 故障在首次新增step之前，与通用“两step”要求存在适用问题 | 请维护者确认初始化故障的接收/验收方式 |
| task.toml/run_spec/verifier | 未生成 | 接收选题与运行契约确认后在官方checkout按规范生成 |
| Solution稳定判分 | 未标定 | 初审建议Issue；若升级，补公平可重复结局和门槛 |
| 开源入库许可 | 上游规范许可已附，本地glue和实验资产没有新授予许可 | 正式发布前与贡献者确认并保留上游许可 |

这些是材料包自身的边界，不是要求维护者相信尚未提供的数据。当前包可直接用于技术初审，不能宣称已经READY合并。

## 核验命令

```bash
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/verify_bundle.py
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/probe_unix_socket.py
```

完整训练的启动命令依赖内部集群、控制与资产，这里没有提供一个会误启动内部任务的伪通用命令。code_context供源码检查，禁止将其单独视为完整可执行训练入口。
