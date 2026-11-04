# 本地 Harbor 盲测

本文件对应主流程阶段 6“检测bench数据”

## 运行准备

- 向用户确认评测模型和judge模型的配置。

- 检查 Python、Harbor、Docker 服务和任务镜像是否可用。
- 显式传入单个任务目录，每轮使用独立运行名和结果目录。

## 执行入口

在本地仓库根目录执行，按下表填写参数。以下为 issue 测试示例：

```bash
python3 -m autorlbench harbor-evaluate tasks/<domain>/<task_id> \
  --evaluation-level issue \
  --judge-model <judge-model> \
  --agent <agent> \
  --model <model> \
  --jobs-dir <run-output-dir> \
  --job-name <unique-run-name>
```

| 参数 | 填写规则 |
| --- | --- |
| `tasks/<domain>/<task_id>` | 填写阶段 5 生成的单个任务目录；`domain` 和 `task_id` 与目标仓库中的实际路径一致。 |
| `--evaluation-level` | 诊断测试填 `issue`，修复测试填 `solution`； |
| `--judge-model` | 填写用户确认的 judge 模型在所用 API 中的模型标识 |
| `--agent` | 填写用户确认的被测 agent 在 Harbor 中的名称。 |
| `--model` | 填写用户确认的被测模型，采用所选 Harbor agent 支持的模型标识格式。 |
| `--jobs-dir` | 填写本机 case 目录下 `checks/harbor/` 的绝对路径； |
| `--job-name` | 每轮填写独立名称，采用 `<task_id>-<evaluation-level>-<时间戳>`，例如 `example-issue-20261027-153000`。 |

## 结果与复核

Harbor 记录位于 `<jobs-dir>/<job-name>/`（即 `checks/harbor/<job-name>/`）：汇总结果见 `result.json` 和 `autorlbench-report.json`，解题轨迹及 verifier 输出位于各 trial 子目录，运行后的 workspace 位于 `<trial>/artifacts/workspace/`。

按以下标准复核 Harbor 实测：

- **仓库没有遗留信息。** Agent 没有利用仓库遗留的答案、参考修复或历史诊断等线索获取结论。
- **Instruction 没有过度暗示。** 任务指令提供必要背景和目标，agent 的解题过程体现自主探索、分析与验证。
- **Workspace 清晰精简。** 目录组织清晰，文件与任务相关，agent 的 token 消耗主要用于理解和解决问题，阅读无关文件的开销较少。

