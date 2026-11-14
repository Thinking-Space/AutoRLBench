
<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img src="assets/logo.png" alt="AutoRLBench" width="360"></a>
</p>

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img alt="Website" src="https://img.shields.io/badge/Website-thinking--space.github.io-4285F4?logo=googlechrome&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-0.1-FFB000?logo=trophy&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#tasks"><img alt="Tasks" src="https://img.shields.io/badge/Tasks-6%20domains-00C853?logo=stack&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/contribute.html"><img alt="Contribute" src="https://img.shields.io/badge/Contribute-a%20run-7B1FA2?logo=git-pull-request&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-D22128?logo=apache&logoColor=white"></a>
  <a href="README_zh.md"><img alt="Chinese (Simplified)" src="https://img.shields.io/badge/Chinese-Simplified-E74C3C"></a>
</p>

A benchmark and open call for valuable, reproducible RL runs. Each task is a real training run that went wrong. The agent has to find the root cause (**issue**) and, where a repaired run exists, fix it and rerun (**solution**).

<p align="center"><img src="assets/ai-trains-ai.jpeg" alt="AI Trains AI" width="480"></p>

## 🚀 News 

- **6 Nov 2026** — We release the second version of the task list: 32 tasks, 24 of which support solution-level evaluation.
- **2 Nov 2026** — We release the first version of the task list: 18 tasks, 13 of which support solution-level evaluation.
- **24 Oct 2026** — Call for contributions officially open, and Kimi-K3.1-Pro added to the leaderboard.
- **19 Oct 2026** — GitHub repo and website released.

## Leaderboard

AutoRLBench Leaderboard, issue + solution evaluation. Overall is the combined pass rate across all evaluated runs (issue + solution): passes over total runs, not an average of the two levels.

| # | Model | Agent | Issue Score | Solution Score | Overall |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | GPT-6-Astra | Codex | 44.1 % | 40.0 % | **42.6 %** |
| 2 | GPT-6.1-Sol | Codex | 38.2 % | 45.0 % | **40.7 %** |
| 3 | Gemini-4-Argon | Claude Code | 23.5 % | 60.0 % | **37.0 %** |
| 4 | GLM-5.3 | Claude Code | 32.4 % | 40.0 % | **35.2 %** |
| 5 | Claude-Fable-5.1 | Claude Code | 26.5 % | 40.0 % | **31.5 %** |
| 5 | Kimi-K3.1-Pro | Claude Code | 23.5 % | 45.0 % | **31.5 %** |
| 7 | Qwen3.8-Max | Qwen Code | 26.5 % | 30.0 % | **27.8 %** |
| 8 | Grok-4.7 | Grok Build | 23.5 % | 25.0 % | **24.1 %** |
| 9 | DeepSeek-V4-Pro | Claude Code | 0.0 % | 15.0 % | **5.6 %** |
| 9 | Kimi-K3 | Claude Code | 0.0 % | 15.0 % | **5.6 %** |

## Tasks

Six domains, one representative task each. The domains come from where in the RL pipeline the failure lives:

<table>
<tr>
<td width="50%" valign="top">
<a href="tasks/configuration/trl_gsm8k_eos_mismatch/">
<img src="assets/task-configuration.jpeg" alt="Configuration" width="100%">
</a>
<strong>Configuration</strong> · issue<br>
<a href="tasks/configuration/trl_gsm8k_eos_mismatch/"><b>TRL GSM8K EOS mismatch</b></a><br>
Missing <code>&lt;|im_end|&gt;</code> in <code>eos_token_id</code> lets rollouts continue past the answer.
</td>
<td width="50%" valign="top">
<a href="tasks/reward/slime_retool_length_reward_bonus/">
<img src="assets/task-reward.jpeg" alt="Reward" width="100%">
</a>
<strong>Reward</strong> · issue · solution<br>
<a href="tasks/reward/slime_retool_length_reward_bonus/"><b>Length bonus reward hack</b></a><br>
An unconditional length bonus teaches the policy to write longer, not better.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/data/verl_taco_difficulty_mismatch/">
<img src="assets/task-data.jpeg" alt="Data" width="100%">
</a>
<strong>Data</strong> · issue<br>
<a href="tasks/data/verl_taco_difficulty_mismatch/"><b>Training data is too hard</b></a><br>
Train is TACO <code>VERY_HARD</code> while validation is <code>EASY</code>; held-out accuracy ends below its start.
</td>
<td width="50%" valign="top">
<a href="tasks/environment/search_r1_zero_query_embedding/">
<img src="assets/task-environment.jpeg" alt="Environment" width="100%">
</a>
<strong>Environment</strong> · issue · solution<br>
<a href="tasks/environment/search_r1_zero_query_embedding/"><b>Zero query vector in retrieval</b></a><br>
The retriever's query vector is all zeros, so every query gets the same documents.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/">
<img src="assets/task-algorithm.jpeg" alt="Algorithm" width="100%">
</a>
<strong>Algorithm</strong> · issue<br>
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/"><b>GRPO uid assigned after rollout repeat</b></a><br>
The uid assignment lands after the rollout repeat step, making every GRPO group a singleton and silently degrading training to baseline-free REINFORCE.
</td>
<td width="50%" valign="top">
<a href="tasks/infrastructure/verl_moe_token_normalization/">
<img src="assets/task-infrastructure.jpeg" alt="Infrastructure" width="100%">
</a>
<strong>Infrastructure</strong> · issue · solution<br>
<a href="tasks/infrastructure/verl_moe_token_normalization/"><b>MoE auxiliary gradients miss token normalization</b></a><br>
The per-token loss callback omits the token count, so MoE z-loss gradients are never normalized.
</td>
</tr>
</table>

## How a task works

```text
prepare ──▶ agent ──▶ [rerun] ──▶ verifier
  │           │          │            │
  │           │          │            └─ issue:    LLM judge on agent_answer.md vs. hidden oracle
  │           │          │               solution: a metric threshold on the rerun's logs
  │           │          └─ solution only: the harness re-runs the experiment on the agent's
  │           │             workspace, per the task's run_spec.json
  │           └─ reads instruction.md and a writable copy of workspace/
  └─ fetches pinned models and datasets (the only stage with network)
```

Each stage runs in its own container from the same prebuilt image. Oracles and verifiers are mounted only into the verifier stage.

At the solution level the agent repairs the workspace and submits `report.md` — it does **not** launch training itself. The harness clears the task's log roots, re-runs the experiment on the repaired workspace, and grades only that fresh run. No `report.md`, no rerun and no grading: the task scores 0. A solution task without a `run_spec.json` (a pure protocol repair, say) skips the rerun but is bound by the same report gate.

## Repository layout

| Path | Contents |
| --- | --- |
| [`tasks/`](tasks/) | Benchmark tasks, grouped by domain |
| [`autorlbench/`](autorlbench/) | Evaluation harness: task loader, validator, runtimes, agent adapters, CLI |
| [`ENVIRONMENTS.md`](ENVIRONMENTS.md) | The eight fixed shared environment images every task runs in |
| [`skills/package-task/`](skills/package-task/) | Agent skill that turns a raw task into a task |
| [`tests/`](tests/) | Harness unit tests |
| [`page/`](page/) | Website source, deployed to GitHub Pages |

## Task layout

```text
tasks/<domain>/<task_id>/
├── README.md               # summary, difficulty, reference solution, verification
├── task.toml               # metadata and environment image
├── workspace/              # the run as the agent sees it: code, config, logs, metrics
├── environment/prepare.sh  # fetches pinned models and datasets (only stage with network)
└── evaluations/
    ├── issue/              # instruction.md · oracle.md · verifier/
    └── solution/           # instruction.md · oracle/ · verifier/ (grader_spec.json · run_spec.json)
```

## Quick start

```bash
# Validate a task
python3 -m autorlbench validate tasks/reward/slime_retool_length_reward_bonus

# Rerun a task's experiment on its original workspace (prepare + run_spec.json,
# no agent, no verifier) — for task submitters to confirm their environment works
python3 -m autorlbench rerun tasks/reward/slime_retool_length_reward_bonus

# Run an agent through Harbor (pip install harbor)
python3 -m autorlbench harbor-evaluate tasks/reward --agent codex --model <model>
```

`harbor-evaluate` records each run under `outputs/harbor-evaluate/<task_id>/<agent>-<model>/<timestamp>/`: the command as an executable `command.sh`, the effective Harbor command, a live `prepare.log` and `harbor.log` (both `tail -f`-able while the run is going), and the final `report.json`. Only stage-start lines go to the terminal; everything else lands in those log files.

`rerun` is the submitter-side environment check: it runs the task's `prepare.sh` and then the experiment its `run_spec.json` declares, on the workspace exactly as the task author left it — no agent, no verifier, no report gate. Each run records itself under `outputs/rerun/<task_id>/<timestamp>/` in the same layout as `harbor-evaluate`: the command as an executable `command.sh`, live `tail -f`-able `prepare.log` and `rerun.log`, the final `report.json`, the experiment's logs under `results/`, and the workspace copy. `passed` in the report means both stages ran to completion, while the experiment's own exit code is recorded but not judged, since a buggy run is expected to fail.

Run `python3 -m autorlbench --help` for every option. The issue verifier calls an OpenAI-compatible judge through `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## Contributing

We welcome both problematic runs and healthy reference runs. Bring the issue materials, or add the repaired run when it exists; an agent does the packaging — inside a clone of this repository, using the [package-task](skills/package-task/SKILL.md) skill. Every task runs in one of the eight fixed shared images in [ENVIRONMENTS.md](ENVIRONMENTS.md): run your buggy run in one of them and ship that run's artifacts — a defect that reproduces in none of the eight may submit its own environment configuration with an explanation instead. Contribute **two valid tasks** and you become a co-author on the benchmark paper. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [contribution guide](https://thinking-space.github.io/AutoRLBench/contribute.html).

## License

Apache-2.0. Each task's workspace keeps the license of its upstream code.
