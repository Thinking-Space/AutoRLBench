# Evidence map
本目录为已核对来源的历史证据导出。

- failure_traceback.log：故障尝试append-only计算日志前230432字节内的一段连续完整异常调用链。没有拼接恢复尝试，也没有重写错误结论。标识已脱敏，路径长度保持150字节。
- failure_summary.json：失败后独立清理确认记录的选取字段；passed=true描述清理成功。
- historical_cpu_acceptance.json / historical_cpu_tests.log：历史45项CPU回归记录及对应原始日志；原测试日志SHA匹配准入凭证。不是本次重跑45项。
- eight_node_cpu_zmq.json：原集群八节点当时的CPU实际ZMQ绑定结果。删除了节点ID和进程FD路径，保留测量值与结果。不是当前监控。
- recovery_restore_and_publication.json：后续真实GPU运行的16rank恢复、优化器/RNG和12副本首次安装的选取字段。gpu_operations=0描述独立核验过程，不代表历史训练未使用GPU。
- recovery_step_16.json / recovery_step_17.json：真实更新、保存、发布的验收字段，含逐rank优化器计数和128条非padding。未重新读取历史checkpoint载荷。

每份JSON都是明确选取的原始字段，来源、删减说明及前后哈希见../provenance/source_manifest.json。它们不冒充原始完整训练日志，实际连续故障traceback和原始metric行另行提供。

../metrics/recovery_steps_16_17.jsonl是原始两行JSONL的精确字节拷贝，原验收中的metrics_row_sha256可直接校对（可能按不含换行计算）。没有改tag、重算、补零或压缩成少数指标。

CPU机制探针是本次新增检查，结果与历史CPU/GPU证据分开保存，不能互相替代。
