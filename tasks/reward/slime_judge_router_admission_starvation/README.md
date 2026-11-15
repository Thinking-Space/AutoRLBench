# Judge service capacity and reward admission disagree in the first GRPO rollout

Training Qwen3-VL-4B-Thinking with GRPO on 768 CUA-Gym tasks starts two Judge replicas but registers one, then rejects 73 of 96 groups by comparing Judge rewards to independent environment scores, accepting only 7 of 32 required groups.

| | |
| --- | --- |
| **Author** | Zhizhou Wang |
| **Affiliation** | Southwest Jiaotong University |
| **Type** | Reward |
| **Domain** | GUI Agent RL |
| **Tags** | `hard-candidate` `judge-reward` `router` `dapo` `multi-cause` |
| **Framework** | slime GRPO, Slime 0.3.0-based runtime |
| **Run time** | ~67 minutes |
| **Resources** | 16 × H100 (80 GB) |

## Difficulty

The service banner claims two workers, policy-router logs initially list none, and long Judge queues coexist with 49 trajectory timeouts. The agent must distinguish the two routers and trace reward provenance across generation and DAPO admission. Source was restored from recorded post-incident edits; the launcher and scheduler line numbers agree with the raw run. Both actual service logs and all 768 compact trajectory records are provided. A low acceptance rate cannot be attributed solely to task difficulty.


Empirical difficulty calibration (2026-11-12): stock dsh-tui with `deepseek-v4.1-flash-mb`, high reasoning, passed **0/2** final isolated attempts (weighted scores **0.266667 / 0.288889**; root/evidence pairs **0.0/0.333333** and **0.0/0.444444**). Both agents completed normally within the 1800-second budget, reported `input_unchanged=true`, and submitted diagnoses; neither result is a timeout or API failure. These two samples establish observed difficulty for this configuration, not universal frontier-model failure.

## Reference solution

- **issue**: A passing diagnosis must identify all three essential causes: the repeated store-action `--worker-urls` argument replaces the first Judge worker so the router registers only `http://host_0:31102`; `dapo.py::episode_rewards` applies an invalid equality check between the Judge reward and independent `gui_score`; and the concrete `generate_with_gui.py` assignment order first stores `eval_score` in `metadata["gui_score"]`, then assigns the Judge verdict returned by `_judge_reward` to `sample.reward["score"]`, leaving the independent sources without an admission-time alignment handoff. Merely observing different logged scores or fields does not establish the third cause. It must also connect the router logs as one evidence chain: `judge-0.log` has startup self-check but no Decode batches, `judge-1.log` carries the workload, and `judge-router.log` registers only one worker URL. Naming only a subset earns partial root-cause/evidence credit. Semaphore queueing, trajectory timeouts, and long Judge outputs are symptoms or amplifiers.

## Verification

- **issue**: the shared LLM judge scores verdict / root cause / evidence with weights 0.2 / 0.6 / 0.2. The current repository verifier requires verdict 1.0 and both root cause and evidence strictly greater than 0.8; the weighted total alone does not determine passing.

Issue-only: no clean same-seed, same-data, same-budget post-fix comparison establishes a reliable repair threshold. Full GPU training is not required for diagnosis and has not been rerun during packaging.

Packaging checks passed: task validator, shared-verifier byte equality, snapshot hashes, and source/shell syntax. A complete reviewer diagnosis scores 1.0; an unsupported learning-rate diagnosis scores 0.2 and fails. The evaluated agent had a fresh isolated filesystem, read-only workspace, writable results, no oracle/reviewer files, no host credentials and no general network. The model-only API broker ran outside the sandbox. This is a CPU diagnosis E2E using the official issue verifier; the declared public container image and full GPU training were not executed in this validation.
