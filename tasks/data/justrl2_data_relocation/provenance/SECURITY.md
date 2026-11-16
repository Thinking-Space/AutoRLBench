# Security scope / 安全边界

## 中文

这是可复现的代码审查/修复题，不是能够抵抗所有恶意系统行为的安全沙箱证明。
测试只覆盖明确的行为契约和已复现的绕过方式；未开展 LLM 盲测。

- Agent 只提交报告和受限的路径查询。固定解释器分析支持的 Python AST，
  不执行提交的任意 Python；文件读取限定在数据目录，限制语法、读取量和操作数。
- Prepare、agent、续训和评分使用分离阶段；准备资产和隐藏判定不由 agent 写入。
  禁止共享进程的本地/Harbor 后端。最终内容核验独立检查 ID、字节哈希及难度。
- 不可用记录探针覆盖不存在的 ID、错误内容哈希，以及旧路径仍存在的容易样本。
  另用临时数据目录检查索引匹配但文件缺失的情况，不修改固定场景。
  参考解对所有样本匹配当前目录中的 ID 和哈希，并检查目标文件存在。
- 评分须结合正确初始判定和新续训证据；单独写成功 JSON、借用旧检查点或
  只交正确报告不能通过。测试覆盖输出/报告软链接与 Python 启动钩子等入口。
- 正确的健康提交无需改代码，但仍须通过完全相同的续训检查。

仍有以下限制：

1. 题库、参考解、固定样本和题目身份公开后可被记忆。中性题面和健康对照
   不能证明排除了数据污染；运行时应隐藏维护者文件并隔离不同题目。
2. 窄的 `loader.py` 接口会提示调查方向，故障机制本身不具有无提示的大型仓库难度。
3. Solution 评分只检查报告有判定和非空正文，不评审正文推理是否正确。
   Issue 评分另行使用语义 judge，两者能力不可混称。
4. 五个不可用记录探针不是所有未来存储状态的穷尽证明。固定场景续训验证
   当前原始样本；新场景或新迁移方式需要独立的契约和验收。
5. 安全结论依赖 Docker 阶段隔离、只读挂载和可信 runner。CPU 测试不替代
   部署环境隔离审计，也不证明 GPU 驱动或容器本身没有漏洞。
6. 现有 GPU 收据早于受限解释器和启动参数加固；本次文件完整性校验和 CPU
   回归不应被表述为加固后的新 GPU 端到端验收或 agent 成功率。

## English

This is a reproducible review/repair benchmark, not a proof against every
adversarial system behavior. Tests cover explicit contracts and reproduced
bypasses; no blind LLM evaluation has been performed.

- Agents submit a report and a bounded path query. The trusted interpreter
  evaluates supported AST forms rather than executing arbitrary submitted
  Python, confines reads to the dataset and limits syntax, bytes and operations.
- Separate prepare, agent, rerun and verifier stages protect prepared inputs and
  hidden metadata. Shared local/Harbor execution is rejected. The trusted outer
  reader independently verifies logical identity, actual bytes and difficulty.
- Unavailable-record probes cover unknown IDs and wrong hashes on both relocated
  and still-existing paths. An isolated dataset also probes valid catalog entries
  with missing backing files, without modifying the pinned scene. The reference
  matches catalog ID/hash for all records and checks the chosen file exists.
- Acceptance combines the correct starting verdict with fresh continuation
  evidence. Preplanted success files, old checkpoints or a report alone cannot
  satisfy it. Tests include report/output symlinks and Python startup hooks.
- Healthy submissions need no diff but must pass the same continuation checks.

Remaining limits:

1. Public answers, fixed samples and task identities can be memorized. Neutral
   instructions and controls do not prove absence of contamination. Hide
   maintainer files at runtime and isolate task instances.
2. The narrow loader interface reveals a likely investigation area. This does
   not have the difficulty of unassisted debugging across a large repository.
3. Solution grading checks verdict and nonempty report text, not reasoning
   quality. The separate issue judge assesses semantics; do not conflate them.
4. Five negative probes do not exhaust future storage states. Continuation
   verifies the fixed original samples. New scenes need separate contracts.
5. The boundary depends on Docker isolation, read-only mounts and a trusted
   runner. CPU regressions do not replace deployment isolation review or prove
   the container/GPU stack is free of vulnerabilities.
6. Existing GPU receipts predate interpreter/launch hardening. File verification
   and CPU checks are not a new hardened GPU acceptance run or an agent success
   rate.
