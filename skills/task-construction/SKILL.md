---
name: task-construction
description: 这是一个存在特定科研用途的 skill，仅当用户在 session 中明确要求使用时触发。基于用户提供的类 issue 信息构造可复现、可判分的 agent 评测数据。
---

# Task Construction

## 文档索引

| 文档 | 内容 | 使用阶段 |
| --- | --- | --- |
| [SKILL.md](SKILL.md) | 总流程 | 任务开始及全程协调 |
| [references/experiments_plan_template.md](references/experiments_plan_template.md) | 实验方案模板 | 阶段 2 生成方案并提供给用户 |
| [references/compute_queue.md](references/compute_queue.md) | 排队脚本使用说明 | 阶段 3 等待空闲算力 |
| [references/benchcase_test.md](references/benchcase_test.md) | 本地 Harbor 测试说明 | 阶段 6 检测 |
| [../package-task/SKILL.md](../package-task/SKILL.md) | Bench 仓库的 package-task 打包流程入口 | 阶段 5 打包；阶段 6 校验 |
| [../package-task/reference.md](../package-task/reference.md) | Bench 任务文件的模板 | 阶段 5 构造任务文件 |
| [../package-task/examples.md](../package-task/examples.md) | 模型、数据、运行和评分参数示例 | 阶段 5 整理打包输入 |
| [../package-task/readme-template.md](../package-task/readme-template.md) | 面向审阅者的任务 README 模板和写作要求 | 阶段 5 编写文档；阶段 7 交付 |

## 基本原则

- **保持本地仓库与远程对齐。** 实时更新用户指定的本地评测仓库（AutoRLBench）状态，在任务开始及交付前同步远程更新。
- **遵守团队开发规范。** 读取并遵循目标仓库的 `AGENTS.md`、贡献约定及校验规范。未经用户明确允许，不得向远程 push。
- **以构建评测数据为复现目标。** 只要真实复现实验的日志记录出现约定的预期异常，即判定复现成功；无需完整重现原错误的全部过程、规模、数值或训练时长。达到取证目标后按约定结束采集，将证据用于任务构造。Task 文档和任务指令不添加针对本次复现的防御性说明。

## 背景说明

将用户提供的类 issue 信息转成 agent 可执行的评测 task。

实际部署为本机运行 Codex、远程服务器执行实验。本机保存方案和维护者记录，通过 SSH 使用远程计算资源。实验数据位于远程服务器，按打包需要将证据同步回本机。

本 skill 与 Bench 仓库的 [package-task](../package-task/SKILL.md) 位于同一 skills 目录。接入 AutoRLBench 时，直接遵循目标仓库的 package-task 打包流程和静态检查要求。本 skill 负责将类 issue 信息整理成所需材料、按授权补齐实验与证据，再按打包流程接入目标仓库。

## 本地工作区目录约定

每个 case 使用独立的本地工作目录，bench 数据直接集成到目标仓库。优先沿用用户指定的位置；以通用的 `localworkdir` 工作区为例：

```text
localworkdir\
├── candidates\                     # 候选问题，与当前任务无关
├── cases\
│   └── <case名称>\
│       ├── case.md                 # 当前阶段、任务路径、远程位置及结果索引
│       ├── sources\                # 用户原始描述、参考资料及原始日志
│       ├── diagnosis.md            # 根因、分类、解决方案和触发环境
│       ├── plan.md                 # 提交用户确认的实验方案
│       ├── environment.md          # 远程主机、目录、缓存、镜像与资源信息
│       ├── queue\                  # 排队状态与 watcher 日志
│       ├── runs\
│       │   └── <variant>\<attempt_id>\
│       │       ├── run.md          # 命令、远程作业、状态和复现结论
│       │       ├── logs\           # 从远程同步的取证日志
│       │       └── artifacts\      # 打包需要的实验产物
│       ├── checks\
│       │   ├── static\             # package skill 检查与 grader 回放结果
│       │   └── harbor\             # Harbor job 输出
│       └── delivery\
│           └── delivery.md         # 交付位置、版本、检测结果与使用入口
├── AutoRLBench-delivery\            # bench 集成与交付仓库
├── AutoRLBench-verify\              # 沿用现有验证 checkout
└── verification\                   # 沿用现有验证材料目录
```

## 工作流

### 1. 确定bug根因和解决方案

#### 根因判断的参考依据

- 用户提供的原始问题描述与预期行为，以及已知根因。
- 对应公开仓库中，issue / PR 给出的诊断信息
- 仓库源码与具体实现。
- 故障版本与正常版本的代码差异。
- 已有运行日志与指标记录。

#### 解决方案的参考依据

- 用户提供的修复线索与已验证解决步骤。
- 对应公开仓库中，issue / PR 给出的解决方案
- 上游修复补丁与正常版本的实现。
- 已有回归测试与故障/正常对照结果。

#### 对根因分类

按根因所在环节选择主类别，使用下表中的六类及对应 AutoRLBench domain。

| 类别 | Domain | 归类依据 |
| --- | --- | --- |
| Configuration（配置） | `configuration` | 配置值或实验协议选择不当，例如评测采样不足或启动参数错误 |
| Reward（奖励） | `reward` | 奖励定义或计算存在问题，例如奖励投机或奖励与任务目标不一致 |
| Algorithm（算法） | `algorithm` | 优化目标或算法实现存在问题，例如优势估计错误或样本分组错误 |
| Infrastructure（基础设施） | `infrastructure` | 底层训练执行存在问题，例如分布式通信错误或并行布局下的算子行为错误 |
| Data（数据） | `data` | 数据内容或处理流程存在问题，例如训练与验证难度不匹配或数据过滤错误 |
| Environment（环境） | `environment` | 任务环境或外部交互组件存在问题，例如检索服务错误或工具接口行为异常 |

#### 确定触发bug的环境

识别触发异常所必需的环境组合，区分已确认的触发条件与普通运行配置。按相关性记录以下信息，未知项标为待核验，并带入阶段 2 的实验设计和阶段 3 的资源检查：

- 故障代码的仓库版本与本地补丁；有正常对照时记录对应版本。
- 容器镜像或操作系统，以及 Python 与关键依赖的实际版本。
- GPU 或其他设备的型号与数量，以及适用的驱动和 CUDA 版本。

环境选择应保留必要触发条件，服务于评测数据构建，无需照搬原运行的全部规模与配置。

用户原始描述、参考资料和原始日志保存到 `sources/`；根因诊断记录到 `diagnosis.md`。

### 2. 确认实验方案

读取 [实验方案模板](references/experiments_plan_template.md)，仿照模板生成针对当前 case 的具体方案，对于无法确认的实验配置、预算和执行范围，通过当前可用的问答工具与用户对齐，优先使用 `request_user_input_async`。已有明确授权时沿用，不重复确认。

选择保留触发条件的最小充分实验。真实运行日志出现约定的预期异常即可判定复现成功。使用本地 wandb 记录训练过程数据。等待必要答复期间继续资料与方案准备，确认后记录方案版本和授权范围。实质改变配置、判据或预算时，确认超出原授权的部分。

将完整实验方案保存到 `plan.md` 并提供给用户确认或修改，确认结果和后续方案调整一并记录。

### 3. 实验准备

按已确认方案准备实验所需资源。

#### 复现目录管理

在本机 `environment.md` 记录远程主机和复现根目录。远程优先沿用用户指定的位置，未指定时在服务器的实验工作区创建对应复现目录。下述源码和运行目录结构位于远程服务器，可采用以下结构：

```text
<repro_root>/<case名称_recreat_experiment>/
├── plan.md                         # 已确认的实验方案
├── src/                            # 实验源码与补丁
├── configs/                        # 公共配置与各组配置
└── runs/<variant>/<attempt_id>/
    ├── command.txt                 # 实际启动命令
    ├── resolved_config.json        # 实际生效配置，格式可随框架调整
    ├── logs/                       # 控制台、指标与 W&B 记录
    └── artifacts/                  # 本轮生成的其他产物
```

模型与数据集缓存尽量使用已有共享目录，记录其实际路径及容器挂载映射；通过配置路径或受支持的挂载方式复用，避免每个 attempt 重复复制。

#### 优先复用本地资源

此处“本地资源”指实验服务器已有的资源。通过 SSH 检查服务器的文件、缓存和镜像；本机已有文件需要复用时，明确上传后的远程路径。

- 数据集：检查本地文件与已有缓存，优先使用满足已确认类型、规模和版本的数据。
- 模型：检查本地权重与模型缓存，优先使用符合方案的模型版本。
- 镜像：检查本地已有容器镜像，优先复用依赖栈匹配的镜像，并记录 tag 及实际 image ID 或 digest。

本地缺少资源时，仅下载缺失部分，优先使用用户指定或当前环境可用的镜像站与镜像源，包括模型/数据下载镜像和容器 registry 镜像，以缩短准备时间。镜像站不可用时可回退到原始源。

将实际资源版本、缓存路径、下载来源及容器挂载映射记录到 `environment.md`。

#### 等待空闲算力与就绪唤醒

资源尚无空闲容量时，按 [算力排队使用说明](references/compute_queue.md) 完成排队等待与就绪唤醒，排队状态和 watcher 日志保存到 `queue/`，资源就绪后进入阶段 4。

### 4. 执行复现实验

通过 SSH 在远程 attempt 目录启动实验，W&B 使用远程离线模式保存到该 attempt 的日志目录。停止或清理作业也在对应远程主机执行。

本机在 `runs/<variant>/<attempt_id>/run.md` 记录实际运行命令，同步的取证日志保存到同目录的 `logs/`，打包所需产物保存到 `artifacts/`。Heartbeat 任务 ID 和监督结果一并记录到 `run.md`。

检查实际日志是否出现预期异常，出现即判定复现成功，并按约定结束取证。需要正常对照时运行参考修复，记录两组的目标差异及结果。失败或出现其他故障时保留证据，按诊断依据和已确认额度处理重试；达到预算边界停止实验并向用户反馈。

#### 通过 heartbeat 监督进程

实验启动并完成启动核验后，agent 调用自动化工具，为当前聊天创建或复用 `kind=heartbeat` 的进程监督任务。默认每 15 分钟唤醒，具体间隔结合运行时长和预算确定。

Heartbeat 沿用本机当前聊天的上下文，提示仅保留一句任务：

> 检查当前远程实验的进程和增量日志，按已确认方案与预算继续处理，无实质变化时保持安静，出现预期异常、失败、完成或需要用户操作时通知，并在实验结束后停用此 heartbeat。

每次唤醒由本机 Agent 通过 SSH 查询对应远程主机，读取远程增量日志，检查最近 step、日志时间、预期异常、设备占用和累计消耗，并更新本机实验状态。连接失败时保留最后观测并记录错误，下一轮重查。状态没有实质变化时保持安静，只在阶段变化、异常、失败、完成或需要用户操作时通知。

### 5. 打包bench数据

正式任务包保存到目标仓库的 `tasks/<domain>/<task_id>/`。

读取目标 Bench 仓库的 [package-task/SKILL.md](../package-task/SKILL.md)，按照该 skill 的要求打包 bench 数据。package-task 随目标仓库维护与更新，直接使用该仓库中的版本。

### 6. 检测bench数据

任务静态检查与适用的 grader 回放按 [Bench 仓库的 package-task](../package-task/SKILL.md) 执行；Harbor 实测按 [benchcase 检测流程](references/benchcase_test.md) 执行。

### 7. 交付

交付说明保存到 `delivery/delivery.md`，所需补丁或归档包放到 `delivery/`，并在 `case.md` 更新最终状态和交付入口。

默认保留本地工作区变更，commit 按用户明确要求执行；未经明确授权不得向远程 push。结束或释放本次资源，更新最终状态与恢复入口，并清理为本任务安排的已完成或取消的定时检查。
