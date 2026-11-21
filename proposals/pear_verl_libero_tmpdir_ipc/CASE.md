# Case study / 案例分析

## 实验与触发条件

任务是通过clean harness运行LIBERO的多轮工具RL工程续训。底层为PEAR / veRL、FSDP2 / SP8 / separate_async，GRPO / DAPO / Score Centering。8节点各8张H20，2节点承担训练（world_size16），6节点承担rollout（12个TP4推理副本）。每次实际更新消费16个有效group×8条轨迹，与上游128×8配方不同。

本次故障发生于CP15恢复后的启动阶段。协调进程在共享目录中使用自己的TMPDIR本身并不是错误；错误是将它不加区分地覆盖到Ray远端worker，而远端worker需要继续使用已有短IPC目录。

## 根因与源码

`continuous_driver.py:_build_configuration_plan` 的 `forwarded` 元组包含 `TMPDIR`、`TEMP`、`TMP`，随后循环生成：

```python
"++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(environment[key])
```

远端worker因此不再继承节点holder提供的短路径。vLLM为engine/core连接创建IPC endpoint时，在长共享目录后追加UUID，得到150字节的Unix socket路径。日志明确报：

```text
zmq.error.ZMQError: ipc path "...[anonymized, length preserved]..." is longer than 107 characters (sizeof(sockaddr_un.sun_path))
```

上面是便于阅读的错误摘要；连续原始调用链位于evidence/failure_traceback.log，实际路径前缀匿名化后仍为150字节。不能把摘要中的省略号用于路径长度复现。

## 主修复

只从远端转发白名单移除三个临时目录变量，让worker继承各自节点的短个人IPC目录；其他需要转发的日志、W&B和命名空间变量保持原列表。完整before/after源码只存在这一处主修改。协调端仍可保留自己的个人TMPDIR，不能误解为“清空所有进程的临时目录设置”。

补丁：patches/01_runtime_env_tmpdir.patch。源码哈希已分别与修复前归档及当时CPU准入凭证核对。对外源码做了节点/地址脱敏，故导出SHA与原件SHA不同；两者都记录在provenance/source_manifest.json。

## 独立列出的恢复修复

首次新更新前失败时，supervisor需要返回该尝试实际启动来源的已验收checkpoint。旧逻辑读取构造出来的旧式launch路径；新逻辑读取该ticket真实的launch.json，并检查ticket、起始step及source_acceptance哈希。

这与Unix socket根因不同，作为同次故障处理的关联修复单独放在patches/02_recovery_binding.patch，**不作为第二个已获接收的任务**。这样可以避免将“推理初始化失败”和“失败后恢复来源验证”混为一个根因。

## 证据分层

| 层次 | 有哪些证据 | 能支持的结论 |
| --- | --- | --- |
| 直接故障日志 | 完整嵌套traceback、150字节endpoint、107限制 | ZMQ bind因路径长度失败 |
| 历史CPU回归 | 45项测试原始日志及准入记录 | 记录的CPU回归通过；不是GPU结果 |
| 原集群CPU检查 | 8节点ZMQ实际bind，73–74字节 | 修复后节点短路径在当时可绑定 |
| 实际GPU恢复 | 16rank参数/Adam/RNG复载；12副本policy15安装 | 恢复与初始权重发布实际发生 |
| 真实后续更新 | 第16/17步验收及原始metric行 | 不止启动成功，后续更新、保存、发布均有记录 |
| 本次新增机制探针 | 本地CPU真实Unix socket长/短地址检查 | 验证内核路径长度机制，不是新RL实验 |

failure_summary.json中passed=true表示“失败后清理独立核验通过”，不表示失败的训练成功。recovery_restore_and_publication.json中的gpu_operations=0是只读验收过程未增加GPU操作，不能解释为历史恢复没有使用GPU。

## 已记录的恢复结果

| 项目 | 第16步 | 第17步 |
| --- | --- | --- |
| 采样policy | 15 | 16 |
| 发布policy | 16 | 17 |
| 有效group | 16 | 16 |
| 每组轨迹 | 8 | 8 |
| 非padding轨迹 | 128 | 128 |
| 合成padding | 0 | 0 |
| 训练world_size | 16 | 16 |
| 物理权重同步秒数 | 73.40110149997054 | 80.83780958596617 |
| token_logp_abs_diff | 0.004620042200096779 | 0.004575731250871387 |

表格来自实际验收和原始metric行，不补造历史数据。逐token差是有效response token上abs(logp_train−logp_rollout)计算后按原生微批/DP口径聚合的标量；不是精确全局token加权均值。log_ppl_diff/log_ppl_abs_diff则先做逐轨迹平均，不能混作上述逐token绝对差。

## 需要排除的误诊

- **GPU OOM**：直接异常为ZMQ IPC路径超限；不能仅凭模型大或服务启动失败就认定OOM。
- **checkpoint损坏**：本案例有原CP15恢复来源，后续同起点成功复载和发布；但这不是所有checkpoint字节的重新独立审计。
- **网络端口冲突**：致命错误指向Unix domain socket文件路径长度，不是TCP端口已占用。
- **训练batch/算法不正确**：故障在首次新增优化器更新前，直接触发链是临时目录转发到ZMQ绑定。
- **看到高利用率就等于rollout正常**：失败后自有guard的利用率不能充当推理/训练证据，本包未用它判定恢复。

## 实验结论的边界

现有材料是一份故障尝试、一次后续恢复及相关检查的工程案例，不是随机化多seed对照，没有固定LIBERO评测提升的结论。选中训练batch reward不是40任务×50初态评测成功率。没有根据这些数值构造Solution通过门槛。

完整共享镜像复现与正式benchmark harness验收仍需单独完成；不能用本次CPU探针、历史单元测试或已有集群的恢复记录替代。
