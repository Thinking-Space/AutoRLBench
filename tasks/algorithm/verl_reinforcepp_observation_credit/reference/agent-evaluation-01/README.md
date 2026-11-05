# SQL debugging-agent result

**The full evaluation ran successfully. The agent did not solve the task: 20/100.**

The agent correctly said training was unhealthy. It blamed bad SQL joins and missed the framework bug that removes reward credit from actions before tool observations. The canonical judge gave verdict/root-cause/evidence scores of 1.0/0.0/0.0. The answer and judge output below are unchanged copies from the run.

## Completed checks

- [x] Ran one fresh Harbor Terminus-2 agent with GPT-5.4 on the prepared historical workspace.
- [x] Graded the answer with the task's unchanged canonical verifier using GPT-5.2.
- [x] Saved the answer, judge result, submitted commands, runtime settings, and provenance.
- [x] Confirmed all 504 original non-model files were unchanged after the run.

## Evidence

| File | Contents |
| --- | --- |
| [agent_answer.md](agent_answer.md) | Exact final answer submitted by the agent |
| [judge_result.json](judge_result.json) | Exact score, pass/fail result, and judge explanation |
| [run_summary.json](run_summary.json) | Evaluated commit, models, limits, resources, timings, and usage |
| [agent_commands.json](agent_commands.json) | All submitted tool calls, in order |
| [harbor_task.toml](harbor_task.toml) | Staged Harbor task configuration |
| [workspace_audit.json](workspace_audit.json) | Post-run comparison with the initial file hashes |
| [canonical_file_hashes.json](canonical_file_hashes.json) | SHA-256 hashes of unchanged instructions, oracle, and verifier |
| [controller-freeze.txt](controller-freeze.txt) | Installed controller package versions |
| [raw_artifact_hashes.json](raw_artifact_hashes.json) | Hashes and sizes identifying separately retained raw logs |

## Settings and interpretation

The evaluated task commit was `4039e6059bfa634032c33c4d7c62285a3229dcdf`. Harbor 0.25.0 ran Terminus-2 with `openai/gpt-5.4`, medium reasoning, a 24-turn/900-second limit, and 4,096 maximum completion tokens per model call. The agent finished after eight episodes, in 65.016 seconds. GPT-5.2 judged the answer. The run used 4 CPUs/16 GiB for the sandbox and 2 CPUs/8 GiB for the controller; no GPUs were allocated. Controller time including setup and grading was 279.237 seconds. See the [Modal run](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-GnKhARZ8nJcju1BUDZJta3).

The actual benchmark Dockerfile supplied the runtime, with tmux/asciinema/ripgrep added for the agent. The task's preparation script populated a dedicated workspace volume. The agent received the canonical instruction and visible workspace, with network access disabled. Reference controls and the hidden oracle were kept outside that volume; Harbor supplied verifier files after the agent finished. Canonical-file hashes match the evaluated commit. A separate live visibility check could not run after the sandbox had shut down; the isolation record is based on the staged inputs and harness configuration.

There was one actual trial, no retry, and no Harbor trial exception. An invalid API-format value and a missing Harbor Modal dependency were fixed during setup before any agent execution. During the trial the agent dumped large files and interrupted heredocs, then completed its answer. Its command trace never inspects the faulty return-estimator implementation. The failed result is retained as observed.

The canonical judge reports a weighted score (verdict/root cause/evidence: 0.2/0.6/0.2). Its pass condition additionally requires verdict score 1.0 and both root-cause and evidence scores strictly greater than 0.8. This trial fails that condition.

The full terminal recordings and Harbor trajectory remain in the author-run artifacts; this compact bundle includes the commands and final result, not the large terminal outputs or private gateway configuration. Harbor estimated $0.161665 for agent model calls only. That excludes the judge, API connectivity checks, preparation, Modal compute, image builds, and storage, and is not a total bill. One trial does not estimate a model's overall success rate.
