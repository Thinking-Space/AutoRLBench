# Security scope / 安全边界

## 中文

本题在 CPU 上验证有限的数据读取契约，不能证明能够抵抗所有恶意系统行为。
当前验收不训练模型，不以奖励改善为条件，也不声称完成 LLM 盲测。

- Agent 提交报告及受限的 `loader.py` 路径查询。固定解释器只求值支持的
  Python AST，不执行任意提交代码；读取限制在数据目录，限制语法、字节量
  与操作数。Agent 不能通过查询替换样本内容或固定读取器。
- 题包的 `provenance/evaluate_submission.py` 评分已有工作区，不启动 LLM
  agent。Prepare 与 verifier 在不同 CPU Docker 容器运行，均禁用网络；
  verifier 只读挂载提交物、准备数据和可信评分代码。保留按 digest 固定的
  共享 miles 镜像，`gpus = 0`；首次获取镜像可能需要网络。
- 上游默认 Harbor 自动 agent 工作流拒绝需要隔离阶段的任务，目前不支持
  本题。独立评分器可评分外部 agent 工作区，但不提供该 agent 的运行沙箱；
  外部运行仍须隐藏维护者判定及答案。参见[交付说明](DELIVERY.md)。
- 固定读取器核对全部原始 64 条训练样本的 ID、实际字节哈希及难度。随后在
  临时副本中将全部训练载荷搬到新生成的不可预知目录，打乱 catalog，在旧
  路径放置错误字节诱饵，再核验同一批样本。硬编码迁移路径不能满足此契约。
- 负例覆盖未知 ID、已有或缺失路径上的错误哈希、catalog 匹配但文件缺失和
  匹配不唯一。参考查询要求 ID 与哈希唯一匹配，并确认目标文件存在；仅判断
  路径存在不能证明内容正确。原始场景不因这些探针而修改。
- 隐藏 verifier 独立执行新检查，忽略预植入的成功 JSON、rerun 结果声明及
  历史日志。通过需要正确初始判定、非空解释和所有行为检查；只写报告或输出
  文件不能代替实际读取。健康实现无需修改，但接受相同检查。

仍有以下限制：

1. 公开答案、固定小样本与题目身份可以被记忆。随机路径及健康对照不能证明
   排除了数据污染。运行时须隐藏维护者材料并隔离各题实例。
2. 窄的 `loader.py` 接口提示了调查范围，不能等同于无提示的大仓库调试难度。
3. Solution 只检查报告判定与非空正文，不验证推理语义；issue judge 独立
   评审诊断，两种评分不可混称。
4. 两种存储布局和这些负例只覆盖固定场景的契约，不穷尽未来迁移、并发更新
   或生产存储故障。随包 16 条评估样本和历史难度标签不是新的模型评估。
5. 安全边界依赖可信 runner、Docker 阶段隔离、正确挂载与固定运行时。
   直接运行 CPU 检查只验证行为，不验证部署隔离或容器本身的安全性。
6. 历史 GPU 收据只描述旧运行；当前 CPU 验收不证明完整 PPO 训练成功、
   GPU 栈可用、奖励改善或 agent 成功率。

## English

This task checks a bounded data-reading contract on CPU. It is not a proof
against every adversarial system behavior. Acceptance does not train a model,
require reward improvement or claim blind LLM testing.

- Agents submit a report and a bounded path query in `loader.py`. The trusted
  interpreter evaluates supported Python AST forms, confines reads to the data
  root and limits syntax, bytes and operations. Submitted queries cannot
  replace payloads or the fixed reader.
- The task-local `provenance/evaluate_submission.py` grades an existing
  workspace; it does not launch an LLM agent. Prepare and verifier run in
  separate CPU Docker containers with networking disabled. Verifier mounts
  for submissions, prepared data and trusted grader code are read-only. The
  shared miles image stays digest-pinned with `gpus = 0`; its first pull may
  require network access.
- The upstream default Harbor automatic agent workflow rejects tasks requiring
  isolated stages and does not support this task. The standalone evaluator can
  grade an external agent's workspace, but does not sandbox that agent's run.
  External runs must still hide maintainer verdicts and answers. See
  [delivery instructions](DELIVERY.md).
- The fixed reader checks the IDs, actual byte hashes and difficulty of every
  original training record. A temporary copy moves all 64 payloads into fresh,
  unpredictable directories, permutes catalog rows and adds wrong-byte decoys
  at stale paths before checking the same samples again. Hardcoded relocation
  paths do not satisfy this contract.
- Negative probes cover unknown IDs, wrong hashes with existing or missing
  paths, matching catalog entries with absent files and ambiguous matches.
  The reference requires a unique ID/hash match and an existing backing file;
  file existence alone does not establish correct content. Probes leave the
  original scene unchanged.
- The hidden verifier independently runs fresh checks and ignores preplanted
  success JSON, rerun success claims and historical logs. Passing requires the
  correct initial verdict, a nonempty explanation and all behavioral checks.
  A report or output file alone cannot substitute for reads. Healthy code may
  remain unchanged but faces the same checks.

Remaining limits:

1. Public answers, a small fixed sample set and task identities can be
   memorized. Random paths and healthy controls do not prove absence of
   contamination. Hide maintainer materials and isolate task instances.
2. The narrow loader interface reveals an investigation area. This does not
   have the difficulty of unassisted debugging across a large repository.
3. Solution grading checks the verdict and nonempty prose, not reasoning
   semantics. The separate issue judge assesses diagnosis.
4. Two storage layouts and these negative probes do not exhaust future
   migrations, concurrent updates or production storage failures. The bundled
   16 evaluation records and historical difficulty labels are not a new model
   evaluation.
5. The boundary depends on a trusted runner, separate Docker stages, correct
   mounts and a fixed runtime. Direct CPU checks establish behavior, not
   deployment isolation or container security.
6. Historical GPU receipts describe earlier runs only. Current CPU acceptance
   establishes neither full PPO success, GPU-stack readiness, reward gains nor
   an agent success rate.
