# RL startup failure caused by remote TMPDIR forwarding
# Ray 临时目录转发导致 vLLM/ZMQ 启动失败

Contributor: 刘江宇  
Prepared: 2026-11-20 (Asia/Shanghai)  
Suggested domain: infrastructure  
Status: **prepared for maintainer review; not yet a validated AutoRLBench task**

## What happened / 案例摘要

在 PEAR / veRL + clean harness / LIBERO 的真实分布式 RL 续训中，协调端把个人共享目录的长路径 TMPDIR、TEMP、TMP 显式转发到 Ray 远端 worker，覆盖了节点预设的短临时路径。vLLM 在该目录下生成 Unix socket，实际路径长 **150 字节**，触发 **107** 字节/字符（本例全 ASCII）限制，初始推理服务启动失败，尚未完成 policy15 首次发布或新的第16步更新。

直接证据见 [failure_traceback.log](evidence/failure_traceback.log)。这是原始 append-only 计算日志中一段连续的完整异常调用链，含 TrainingRunner → PearVLLMServer → vLLM engine/core client → ZMQ bind 的嵌套异常；来源字节范围、行号和原件/导出 SHA-256 见 [source_manifest.json](provenance/source_manifest.json)。目录前缀已经匿名化，**保留原始路径字节长度**。

The remote Ray environment overrode a node-local short temporary path with a long canonical shared-storage path. A 150-byte vLLM IPC endpoint exceeded the 107-byte limit and prevented rollout initialization before the next optimizer update.

## What is in this package / 附件内容

| 内容 | 位置 |
| --- | --- |
| 可读案例、因果链与容易误判的解释 | [CASE.md](CASE.md) |
| 修复前/后两份调度源码（供审阅，不是完整训练框架） | [code_context/README.md](code_context/README.md) |
| 主问题的最小补丁：不再转发远端临时目录 | [01_runtime_env_tmpdir.patch](patches/01_runtime_env_tmpdir.patch) |
| 单独列出的失败恢复路径修复，避免混淆根因 | [02_recovery_binding.patch](patches/02_recovery_binding.patch) |
| 真实故障 traceback、历史45项CPU测试日志、八节点ZMQ绑定结果 | [evidence/README.md](evidence/README.md) |
| 实际恢复/首次权重发布、16/17步更新保存发布记录 | [恢复记录](evidence/recovery_restore_and_publication.json) |
| 第16/17步原始指标行：每行98个tag，数值与原行不改动 | [recovery_steps_16_17.jsonl](metrics/recovery_steps_16_17.jsonl) |
| 故障/恢复配置摘录、固定源码/模型版本与环境边界 | [ENVIRONMENT.md](ENVIRONMENT.md) |
| 新增CPU机制检查与自包含证据校验 | [repro/README.md](repro/README.md) |
| 逐文件SHA-256、脱敏规则与来源 | MANIFEST.json、[provenance/](provenance/source_manifest.json) |

## Observed recovery / 已有实际恢复证据

- 修复后历史45项CPU回归通过，八节点各一次真实CPU ZMQ bind 均通过，测得 socket 路径73–74字节。
- 后续真实运行从 CP15 复载16个训练rank的参数、优化器计数与RNG，并在12个推理副本完成policy15物理发布；同步耗时54.602446810982656秒。
- 继续完成第16、17步真实更新、checkpoint保存与policy16/17发布；两步均为16组×8条，128条非padding、0合成padding。
- 第16/17步的 `actor/rollout_corr/token_logp_abs_diff` 分别为0.004620042200096779、0.004575731250871387。指标口径见 CASE.md；它们只用于证明原始训练指标完整存在，不作为任务成功率或修复收益指标。

以上是历史运行证据。本次整理没有重跑GPU训练，也没有重新打开或复制模型/checkpoint载荷。一次故障尝试和一次后续恢复运行不构成多seed收益实验。

## Review request / 希望维护者确认

优先按 **Issue / infrastructure** 候选初审。我们希望确认：
1. 首次新增训练更新之前的初始化故障是否适合收录，以及如何适用“startup rerun至少两个真实step”的通用门槛。
2. 移植到指定共享镜像时，完整多机RL实验和CP15恢复来源的推荐最小交付方式。

当前仍未完成指定八镜像之一中的故障复现、完整自包含RL workspace、正式task.toml/run_spec/verifier及AutoRLBench评测。不存在已经通过其validator/review的声明，也没有预设Solution阈值。详细缺口见 ENVIRONMENT.md；CPU机制探针不替代RL验收。

## Local inspection / 本地查看

使用Python 3.9+，不需要GPU、Ray、vLLM或联网：

```bash
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/probe_unix_socket.py
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/verify_bundle.py
```

第一条只在本包自己的repro目录创建并清理临时socket；第二条只读证据与校验清单。它们不会导入或执行code_context中的生产调度模块。

The CPU probe demonstrates the Unix-socket path-length mechanism only. The verifier checks the exported evidence, original metric-row hashes, source patches and manifest. Neither command launches RL training.

公开规范快照与链接见 [references/](references/source_index.json)。代码与证据的使用范围见 [RIGHTS.md](RIGHTS.md)。
