# AReaL tau2 user tool history disappears

AReaL GRPO rollout evaluation with Qwen3-4B and Qwen3-8B on telecom erases the user simulator's tool history, producing repeated tool use and zero reward.

| | |
| --- | --- |
| **Author** | AutoRLBench contributors |
| **Domain** | environment |
| **Tags** | `areal` `tau2` `telecom` `tool-use` `grpo` |
| **Model** | Qwen3-4B agent + Qwen3-8B user simulator |
| **Dataset** | tau2-bench telecom, 4 tasks x 4 seeds |
| **Framework** | AReaL, `launch.py` |
| **Environment** | `areal-tau2` · 1×A800 80 GiB, two independent services |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 control calibration collection, plus 1 packaged control replay; 16 trajectories each |

## Difficulty

`areal/examples/tau2/agent.py` enables `_clean_messages(for_user=True)` for simulator requests. It deletes tool results and assistant tool calls after `UserState.flip_roles()` has already established simulator ownership. The preserved telecom user tools make repeated calls possible while the shared 200-step budget keeps shrinking.

| Recorded measure | Defective | Control |
| --- | --- | --- |
| Error-free trajectories / attempted | 15 / 16 | 15 / 16 |
| Tool-history retention | 0.0 | 1.0 |
| Mean reward, error-free trajectories | 0.000000 | 0.133333 |
| Mean reward, all attempts with errors scored zero | 0.000000 | 0.125000 |
| Mean agent turns, error-free trajectories | 10.40 | 11.67 |
| Mean user turns, error-free trajectories | 65.40 | 9.33 |

Each collection has one separately recorded context-length error. The request excerpts establish history erasure in successful executions of the runner. The 42,413 incoming result occurrences are cumulative history observations, with earlier turns counted again on subsequent requests.

## Reference solution

- **issue** diagnoses the user-side branch, both erasures, and the simulator perspective established by `flip_roles`. Reward strictness, learning-rate changes, and the separate context-length error do not explain the lost request history.
- **solution** preserves the simulator's tool-call/result pairs and retains null-field cleanup and input immutability. The reference production patch is AReaL PR #1759. [Upstream repair](https://github.com/areal-project/AReaL/pull/1759)

## Verification

- **issue** uses the shared LLM judge with verdict/root cause/evidence weights 0.2/0.6/0.2. Passing requires verdict 1.0 and both root-cause and evidence scores strictly above 0.8.
- **solution** requires `report.md`, then the harness runs `python3 launch.py` for 16 trajectories with at least 12 error-free episodes. Fresh `telecom/user_tool_history_retention` must be in [0.99, 1.0] and pass hidden completion-entry compatibility checks. The two calibration margins are 0.99 and 0.01. Completion and positive user-tool traffic are checked independently. One paired collection supports these observed margins.

The task image has been built and the prepare script has completed against the pinned cached assets. Local grader replay rejects the defective arm and an identity-cleaner bypass, and accepts the reference. The packaged-entry replay completes 16/16 error-free trajectories, retention 1.0, mean reward 0.1875, and passes the fresh-run grader. Harbor issue testing with Terminus-2 and DeepSeek-V4.1-Flash-1017 scores 0.90 and fails the standard evidence gate, with verdict/root-cause/evidence scores 1.0/1.0/0.5. Solution blind evaluation was cancelled at the user's request.
