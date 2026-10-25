

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img src="assets/logo.png" alt="AutoRLBench" width="360"></a>
</p>

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img alt="Website" src="https://img.shields.io/badge/Website-thinking--space.github.io-4285F4?logo=googlechrome&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-0.1-FFB000?logo=trophy&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#tasks"><img alt="Tasks" src="https://img.shields.io/badge/Tasks-6%20domains-00C853?logo=stack&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/contribute.html"><img alt="Contribute" src="https://img.shields.io/badge/Contribute-a%20run-7B1FA2?logo=git-pull-request&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-D22128?logo=apache&logoColor=white"></a>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-README-3498DB"></a>
</p>

一个面向真实、可复现 RL 训练运行的 benchmark，也是一个开放征集。每个任务都来自一次真实跑坏的训练：agent 要找到根因（**issue**），有修复版本的任务还要改好并重跑（**solution**）。

<p align="center"><img src="assets/ai-trains-ai.jpeg" alt="AI Trains AI" width="480"></p>

## Leaderboard

AutoRLBench Leaderboard 0.1，issue 档成绩。*初版，会随新 agent 与新任务持续更新。*

| # | 模型 | Agent | Issue 分数 | |
| ---: | --- | --- | ---: | :--- |
| 1 | Claude-Fable-5.1 | Claude Code | 0.86 | ██████████████████░░ |
| 2 | Grok-4.7 | Grok Build | 0.79 | ████████████████░░░░ |
| 3 | GPT-6-Astra | Codex | 0.64 | █████████████░░░░░░░ |
| 4 | GLM-5.3 | Claude Code | 0.64 | █████████████░░░░░░░ |
| 5 | GPT-6.1-Sol | Codex | 0.57 | ████████████░░░░░░░░ |
| 6 | Qwen3.8-Max | Qwen Code | 0.57 | ████████████░░░░░░░░ |
| 7 | Kimi-K3 | Claude Code | 0.50 | ██████████░░░░░░░░░░ |
| 8 | Gemini-4-Argon | Claude Code | 0.43 | █████████░░░░░░░░░░░ |
| 9 | DeepSeek-V4-Pro | Claude Code | 0.29 | ██████░░░░░░░░░░░░░░ |

solution 档（修复并重跑）即将开放。完整结果见[网站](https://thinking-space.github.io/AutoRLBench/#leaderboard)。

## 任务

六个 domain，每个放一个代表性任务。domain 按失败发生在 RL 流程的哪一环来划分：

<table>
<tr>
<td width="50%" valign="top">
<a href="tasks/configuration/verl_aime25_single_sample_protocol/">
<img src="assets/task-configuration.jpeg" alt="Configuration" width="100%">
</a>
<strong>Configuration</strong> · issue · solution<br>
<a href="tasks/configuration/verl_aime25_single_sample_protocol/"><b>AIME25 单样本方差</b></a><br>
每题只采样一次，30 题的分数噪声大到无法判断这次运行。
</td>
<td width="50%" valign="top">
<a href="tasks/reward/slime_retool_length_reward_bonus/">
<img src="assets/task-reward.jpeg" alt="Reward" width="100%">
</a>
<strong>Reward</strong> · issue · solution<br>
<a href="tasks/reward/slime_retool_length_reward_bonus/"><b>长度奖励劫持</b></a><br>
无条件长度奖励让策略学会写得更长，而不是写得更好。
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/data/verl_taco_difficulty_mismatch/">
<img src="assets/task-data.jpeg" alt="Data" width="100%">
</a>
<strong>Data</strong> · issue<br>
<a href="tasks/data/verl_taco_difficulty_mismatch/"><b>训练数据太难</b></a><br>
训练集是 TACO <code>VERY_HARD</code>、验证集是 <code>EASY</code>，held-out 准确率跌回起点以下。
</td>
<td width="50%" valign="top">
<a href="tasks/environment/search_r1_zero_query_embedding/">
<img src="assets/task-environment.jpeg" alt="Environment" width="100%">
</a>
<strong>Environment</strong> · issue · solution<br>
<a href="tasks/environment/search_r1_zero_query_embedding/"><b>检索查询向量全零</b></a><br>
检索器的 query 向量全是零，任何问题都返回同一批文档。
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/">
<img src="assets/task-algorithm.jpeg" alt="Algorithm" width="100%">
</a>
<strong>Algorithm</strong> · issue<br>
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/"><b>GRPO uid 被移到 repeat 之后</b></a><br>
uid 赋值被挪到 rollout repeat 之后，每个 GRPO 组退化为单元素，训练静默降级为无基线的 REINFORCE。
</td>
<td width="50%" valign="top">
<a href="tasks/infrastructure/verl_moe_token_normalization/">
<img src="assets/task-infrastructure.jpeg" alt="Infrastructure" width="100%">
</a>
<strong>Infrastructure</strong> · issue · solution<br>
<a href="tasks/infrastructure/verl_moe_token_normalization/"><b>MoE 辅助梯度缺少 token 归一化</b></a><br>
per-token loss 回调缺少 token 计数，MoE z-loss 梯度从未被归一化。
</td>
</tr>
</table>

## 一个任务怎么跑

```text
prepare ──▶ agent ──▶ [rerun] ──▶ verifier
  │           │          │            │
  │           │          │            └─ issue:    LLM judge 比对 agent_answer.md 与隐藏 oracle
  │           │          │               solution: 按指标门槛判定重跑产出的日志
  │           │          └─ 仅 solution：harness 按任务的 run_spec.json，
  │           │             在 agent 修好的 workspace 上重跑实验
  │           └─ 读取 instruction.md 与一份可写的 workspace/
  └─ 拉取钉死版本的模型与数据集（唯一联网的阶段）
```

各阶段各自起容器，用的是同一个预构建镜像。oracle 与 verifier 只挂进 verifier 阶段。

solution 层里 agent 只负责修 workspace 并提交 `report.md`，**不自己启动训练**。harness 会清掉任务声明的日志目录，在修好的 workspace 上重跑实验，只对这一遍的新日志判分。没交 `report.md` 就不重跑也不判分，直接记 0 分。没有 `run_spec.json` 的 solution 任务（比如纯协议修复）不触发重跑，但同样受 report 门槛约束。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`tasks/`](tasks/) | 按 domain 分组的 benchmark 任务 |
| [`autorlbench/`](autorlbench/) | 评测框架：任务加载、校验、运行时、agent 适配器、CLI |
| [`docker/`](docker/) | 共享环境镜像与容器内工具（[说明](docker/README.md)） |
| [`skills/package-task/`](skills/package-task/) | 把 raw task 打包成任务的 agent skill |
| [`tests/`](tests/) | 框架单元测试 |
| [`page/`](page/) | 网站源码，部署到 GitHub Pages |

## 任务结构

```text
tasks/<domain>/<task_id>/
├── README.md               # 摘要、难度、参考解、判分方式
├── task.toml               # 元数据与环境镜像
├── workspace/              # agent 看到的那次运行：代码、配置、日志、指标
├── environment/prepare.sh  # 拉取钉死版本的模型与数据集（唯一联网阶段）
└── evaluations/
    ├── issue/              # instruction.md · oracle.md · verifier/
    └── solution/           # instruction.md · oracle/ · verifier/（grader_spec.json · run_spec.json）
```

## 快速开始

```bash
# 校验一个任务
python3 -m autorlbench validate tasks/reward/slime_retool_length_reward_bonus

# 在 Docker 里运行 prepare 与 verifier（不含 agent）
python3 -m autorlbench evaluate tasks/reward/slime_retool_length_reward_bonus \
  --evaluation-level issue --backend docker \
  --judge-model <judge-model> --json

# 通过 Harbor 运行 agent（pip install harbor）
python3 -m autorlbench harbor-evaluate tasks/reward --agent codex --model <model>
```

全部参数见 `python3 -m autorlbench --help`。issue 档的 verifier 通过 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 调用 OpenAI 兼容的 judge。`evaluate` 不跑 agent，所以 `--evaluation-level solution` 会停在 report 门槛记 0 分，除非 workspace 里本来就带了 `report.md`。

## 贡献

有问题的 run 和健康的对照 run 都欢迎。带上 issue 所需材料即可，已有修复版本时再补上 solution 运行；打包由 agent 完成。贡献**两个有效任务**即可成为 benchmark 论文的共同作者。详见 [CONTRIBUTING.md](CONTRIBUTING.md) 与[贡献指南](https://thinking-space.github.io/AutoRLBench/contribute.html)。

## 许可

Apache-2.0。各任务的 workspace 沿用其上游代码的许可。
