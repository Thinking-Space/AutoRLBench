# Canonical issue oracle

Expected verdict: **unhealthy**.

## Root cause

Causes 1, 2a, and 2b are all essential. Identifying only part of them earns partial
`root_cause_score`; a generic statement that the Judge is slow or that rewards are
inconsistent does not establish the separate code defects.

### 1. The Judge router receives only the last worker URL

`workspace/src/gui_actor_rl/scripts/start_reward_services.sh` appends the
store-action `--worker-urls` option once inside the loop over Judge replicas.
The router parser treats this list-valued option as a single value, so the
second occurrence replaces the first and only `http://host_0:31102` is
registered. Expected: pass one option containing both worker URLs, or otherwise
register both replicas. This is the Judge router; the policy router's initial
empty worker list is a separate, intentional dynamic-registration mode.

### 2a. `episode_rewards` applies an invalid equality check

`workspace/src/gui_actor_rl/dapo.py::episode_rewards` rejects a trajectory as
`inconsistent_reward` when `child.get_reward_value(args)` is not exactly equal
to `child.metadata['gui_score']`. The equality is not a valid invariant for
Judge mode: the configured training reward is the Judge outcome, while
`gui_score` is an independent environment evaluation. The expected check is
that the configured reward is present, valid, and internally consistent with
its own source; it must not require equality with the independent score.

### 2b. The reward-producing path never aligns the two sources before admission

This is a separate data-flow requirement from the equality branch in 2a. In
`workspace/src/gui_actor_rl/generate_with_gui.py`, `env_client.evaluate()` runs
first and produces `eval_score`; sample finalization then stores that value in
`sample.metadata["gui_score"]`, while the later `_judge_reward(...)` call returns
the Judge verdict and assigns it to `sample.reward["score"]`. The production path intentionally keeps these as independent monitoring and training fields, but the admission path consumes them under an equality invariant without a source-aware handoff. A diagnosis must
explicitly identify both sides of this actual order: finalization writes
`sample.metadata["gui_score"]` from `eval_score`, then `_judge_reward` supplies
`sample.reward["score"]`, with no alignment handoff between the independent
sources (or give an equivalent concrete code path). Merely observing that two
logged numbers, fields, or reward names differ does not establish 2b. The
resulting disagreement is then rejected by 2a. Fixing only the router
registration leaves this reward-source split intact, and fixing only the
comparison leaves the router at one worker.

## Observable impact

A complete diagnosis should connect these grouped facts from the frozen run:

1. The service log advertises two Judge replicas, but `judge-0.log` has only its startup self-check and no Decode batches; `judge-1.log` carries the workload, while `judge-router.log` registers exactly one worker URL (`http://host_0:31102`). Its later `workers: []` line does not repair that startup admission.
2. `rollout_0000_dapo.json` examined and submitted 96 groups and consumed a 96-group prefix, but accepted only 7 against the target of 32.
3. The same audit records 73 `inconsistent_reward`, 13 `invalid_episode`, and 3 `zero_or_low_std` rejections; acceptance rate and submitted utilization are both about 0.0729, `batch_ready_s = -1`, and the error reports DAPO exhaustion after 96 candidates.
4. No optimizer update follows this failed audit.
5. `runtime_config.yaml` selects `gui_reward_source: "judge"`.
6. `task_history.jsonl` records separate `env_score`, `judge_status`, and `reward_score` fields, including valid completed/truncated trajectories whose Judge reward differs from the environment score.

## Competing explanations

The model is not simply faced with uniformly zero-variance tasks. Environment evaluate latency, long Judge outputs, semaphore queueing, trajectory timeouts, and parse failures do not explain the repeated router option or the reward-source mismatch. The policy router's initial empty worker list is not this registration defect. No quantitative claim of total rollout speedup is required or established by this snapshot.
