# reference — harbor-evaluate deep dive

Beyond the operational flow in SKILL.md, this file holds: file-by-file run directory
layout, harness internals, the env contract, result field semantics, anti-leak and
anti-cheat essentials, a full debug table, and the task inventory. The authoritative
reference for CLI arguments is `python3 -m autorlbench harbor-evaluate --help`.

## 1. Run directory layout (file by file)

One `harbor-evaluate` invocation leaves a self-contained record under
`outputs/harbor-evaluate/<task_id>/<agent>-<model>/<timestamp>/` (`--output-root` changes
the root; `<agent>-<model>` is parsed from the forwarded `--agent/--model`, default
`default`):

```text
<timestamp>/
├── command.sh                  # the autorlbench invocation verbatim (executable; replay entry point)
├── harbor-command.sh           # the harbor run command actually executed (secrets redacted)
├── prepare.log                 # live output of the prepare stage (tail -f friendly)
├── harbor.log                  # mirror of the harbor CLI console (sparse)
├── run.log                     # stage lines + final "[RESULT] passed=.. score=.. trials=.."
├── report.json                 # full report (all trials; the CLI also prints it when it exits)
└── harbor-job/                 # harbor's job directory (jobs-dir is this run directory)
    ├── job.log                 # global job progress
    ├── logs/agent/<agent>.txt  # the agent's live activity (streaming JSON)
    └── <trial>/                # harbor-generated trial directory
        ├── result.json                        # harbor trial result
        ├── report.json                        # this trial's EvaluationResult (read this for the summary)
        ├── autorlbench-harbor-command.json
        ├── trial.log                         # trial summary log
        ├── verifier/autorlbench-result.json  # the verifier's raw grading (issue judge / solution grader)
        ├── verifier/autorlbench-run.log       # (solution) the training log of the system re-run
        ├── artifacts/workspace/               # the agent's final workspace (large-file filtered)
        └── artifacts/logs/artifacts/          # collected from the container's /logs/artifacts
            ├── agent_answer.md                # (issue) the deliverable
            ├── report.md                      # (solution) the deliverable
            └── experiment_logs/ ...           # (solution) the fresh run the grader scores
```

Note: when multiple tasks are passed in one batch, each task is still its **own job**, but
the final CLI prints only the merged report; per-task details live in each run's own
`report.json`.

## 2. What the harness does behind the scenes

**materialize (generated per evaluation, deleted afterwards)**: translates the task
package into a harbor task — `instruction.md`, `environment/` (prepare.sh), `tests/`
(oracle + verifier), `solution/` (the oracle solution; only uploaded for
`harbor run -a oracle`, never visible to the agent), and the generated `task.toml`
(agent/verifier stages, compose overlay, timeout budgets).

**Task-id anti-leak**: inside the agent container `AUTORLBENCH_TASK_ID` is always the
literal `"task"` (many task ids spell out the diagnosis, e.g. `context_parallel_oom`);
only the verifier container gets the real id. The `environment/` directory is staged
into a digest-named directory before it is bind-mounted (mount sources are readable
from `/proc/mounts` inside the container), so no repo path carrying the task id ever
appears in a container-visible mount. **So evaluators do not need to
scrub task names themselves** — but new tasks added under `tasks/` must still go through
package-task's sanitization flow.

**The verifier wrapper (the core of the solution level)**: one script chaining the
report gate → system re-run → grader: no non-empty `report.md` means an immediate zero
and nothing else runs; otherwise it clears the task's declared log roots (even if the
agent cheated and ran training itself, those logs do not count), re-runs the experiment
on the agent's repaired workspace per `run_spec.json`, then runs
`bash $AUTORLBENCH_VERIFIER/test.sh` to grade; if the rerun exits non-zero, diagnostics
are kept but the score is forced to 0.

**Issue level does not re-run**: inside the verifier container only `test.sh` →
`judge.py` runs, reading `agent_answer.md` against the hidden `oracle.md` via an LLM
judge call.

**Timeout budgets**: the verifier stage cap = `run_spec.timeout_s` + 30 minutes of
grading headroom; the prepare stage is estimated from the upload size of `environment/`
(30 minutes base). The old problem of hitting harbor's default 600s is handled by the
generator — no manual tuning needed.

**GPUs**: the task's `task.toml` `environment.gpus` drives the compose overlay's device
reservations and the raised `/dev/shm` (needed by verl/ray/NCCL). Without enough local
GPUs, solution tasks with a re-run cannot run; the issue level never needs GPUs.

## 3. Env contract

harbor expands `${VAR}` templates from the process environment (including the repo-root
`.env`):

| Variable | Who reads it | Notes |
| --- | --- | --- |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | issue judge | OpenAI-compatible gateway; put in `.env`, auto-loaded by harbor |
| `AUTORLBENCH_JUDGE_MODEL` | issue judge | injected into the verifier stage by `--judge-model` |
| `AUTORLBENCH_JUDGE_TIMEOUT_S` | issue judge | 60s default inside the container; the local re-judge (SKILL §2.4) for a slow judge is not bound by it |
| `<per-agent keys>` | the chosen agent | the agent CLI runs in-container; credentials injected by harbor (e.g. `ANTHROPIC_API_KEY`) |
| `AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_RESULTS` | both stages | always `/workspace` / `/logs/artifacts`; where the agent writes deliverables |

**Deliverable paths**: issue → `$AUTORLBENCH_RESULTS/agent_answer.md`; solution →
`/workspace/report.md` (missing/empty = score 0) + the fresh run's logs under
`$AUTORLBENCH_RESULTS/experiment_logs`.

## 4. Result field semantics

Key fields of `report.json` (and each trial's `report.json`):

| Field | Semantics |
| --- | --- |
| `passed` | grader verdict; **always false when infrastructure_errors is non-empty** — not a capability conclusion |
| `score` | grader score; forced to 0.0 on infra errors |
| `metrics.status` | issue-only: `"judged"` = valid; `"unavailable"` = the judge call failed (carries `error`), must be re-judged |
| `metrics.verdict_score / root_cause_score / evidence_score / reason` | issue judge sub-scores and rationale |
| `metrics.failure` | solution-only: `"agent did not submit report.md"` / `"system rerun failed"` (capability-class true zeros) |
| `metrics.infrastructure_errors` | non-empty = invalid run (harbor exit≠0 / trial exception / workspace collection failure) |
| `metrics.harbor_rewards.reward` | harbor's native reward channel, same source as score |
| `workspace_before_hash` / `workspace_after_hash` | pre/post tree digests, useful to tell whether the agent changed anything |

Issue total = weighted sum with task-defined weights (e.g. 0.2v+0.6r+0.2e; **per-task —
check the `WEIGHTS` header of that task's `evaluations/issue/verifier/judge.py`**); the
pass rule is uniformly `verdict==1.0 AND root_cause>0.8 AND evidence>0.8`.

`solution` grading protocol: `score_mode=final` (the declared step count's validation
metric, no best-of-N; `missing finite <metric> at step N` = never reached the declared
step count, recorded as 0) + validation-first (training reward is recorded but never
graded). Details in the task's
`evaluations/solution/verifier/resources/grader_spec.json`.

## 5. Anti-leak and anti-cheat (two evaluator disciplines)

1. **Do not do the agent's work for it**: no hints dropped into the workspace, no
   pre-running experiments, no editing task packages. The only evaluation write locations
   are `outputs/` (automatic) and your own aggregation artifacts.
2. **Post-run spot check for solutions**: for passing trials, `diff -r` the original
   `workspace/` against `artifacts/workspace/` and confirm the change surface matches what
   `report.md` claims; scan the agent log for network forensics (`git clone` / `curl` /
   `pip install` to external sources) — task protocols generally forbid networking, so
   fixes that "pass" by pulling upstream sources are invalid; flag them in the aggregate.

## 6. Full debug table

| Symptom | Cause | Action |
| --- | --- | --- |
| issue `status: "unavailable"` | judge API failure (timeout/key/gateway) | re-judge locally only (SKILL §2.4); do not re-run the agent |
| `infrastructure_errors` contains "Harbor command failed" | harbor itself failed | see the top of `run.log` and the earlier raise in `harbor.log` |
| `infrastructure_errors` contains "Harbor trial exception" | the trial raised | see `<trial>/trial.log` to locate the stage |
| `infrastructure_errors` contains workspace artifact collection failure | big files blew the 120s collection budget | suffix-based filtering already exists; new file types require trimming the workspace (a task-package change — ask the maintainers) |
| prepare sits still for a long time | pulling models/datasets | normal; `tail prepare.log`; if slow, add `--huggingface-mirror` |
| `missing or invalid agent_answer.md` | the agent wrote no deliverable | capability class; re-run within quota (max 3); first check `logs/agent/<agent>.txt` to confirm it was really working |
| solution `"agent did not submit report.md"` | report gate | a true zero; record as a capability failure |
| solution `"system rerun failed"` | the re-run crashed (OOM etc.) | see `verifier/autorlbench-run.log` and its `rerun exit=` line; capability failure, record 0 |
| containers won't start / image missing | image not pulled | `python3 -m autorlbench build-images <stack>` (list under `docker/images/`) |
| want a shell inside the task environment | — | `python3 -m autorlbench harbor-shell <task> --evaluation-level <level>` (same container shape, no agent) |
| want to verify the task environment itself is healthy | — | `python3 -m autorlbench rerun <task>` (maintainer-side self-check, unrelated to model evaluation; use it when triaging environment problems) |
| the agent is completely idle | keys/gateway | check `.env`; `logs/agent/<agent>.txt` usually shows 401/timeout traces |

## 7. Appendix A: task inventory (32 tasks; ✓ = has evaluations/solution)

| Domain | Task | sol |
| --- | --- | :-: |
| algorithm | maestro_grpo_reward_collapse | ✓ |
| algorithm | trl_dapo_accumulation_normalization | ✓ |
| algorithm | verl_grpo_advantage_broadcast | ✓ |
| algorithm | verl_grpo_uid_after_repeat | |
| algorithm | verl_pissa_residual_base_reference | ✓ |
| algorithm | verl_policy_loss_padded_mask_only | ✓ |
| algorithm | verl_policy_loss_padded_unmasked | ✓ |
| algorithm | verl_reinforcepp_observation_credit | |
| configuration | trl_gsm8k_eos_mismatch | |
| configuration | verl_grpo_adamw_eps_too_large | ✓ |
| configuration | verl_grpo_ppo_mini_batch_too_small | ✓ |
| configuration | verl_grpo_rollout_top_p_off_policy | ✓ |
| data | justrl2_data_relocation | ✓ |
| data | qwen25_gsm8k_grpo | |
| data | verl_taco_difficulty_mismatch | |
| environment | search_r1_zero_query_embedding | ✓ |
| infrastructure | miles_async_rollout_inflight_not_evicted | ✓ |
| infrastructure | qwen3vl_geo3k | ✓ |
| infrastructure | r1d15_math_grpo | ✓ |
| infrastructure | slime_critic_value_head_random_init | ✓ |
| infrastructure | torch_functional | ✓ |
| infrastructure | trl_tool_history_cross_contamination | |
| infrastructure | verl_checkpoint_host_memory_oom | |
| infrastructure | verl_lora_merge_base_weight_sync | ✓ |
| infrastructure | verl_moe_token_normalization | ✓ |
| infrastructure | verl_pissa_vllm_adapter_request | ✓ |
| infrastructure | verl_qwen3_vl_grpo_training_case_v8k2 | |
| infrastructure | verl_rollout_weight_overwrite | ✓ |
| reward | idea_awm_reward_diagnosis | ✓ |
| reward | slime_retool_length_reward_bonus | ✓ |
| reward | slime_search_r1_nomask | ✓ |
| reward | verl_math_verify_thread_zero_reward | |

Trust `ls tasks/<domain>/` over this table (update it when tasks are added/removed).

## 8. Appendix B: agent configuration quick reference

The agent runs its own CLI inside the harbor container; `--agent`/`--model` are
forwarded to `harbor run` as-is, with the common extra `--agent-kwarg key=value` (agent
config items, e.g. `reasoning_effort=max`). Selection conventions:

| Model family | `--agent` | Notes |
| --- | --- | --- |
| Claude-family / most gateway models | `claude-code` | default choice; credentials `ANTHROPIC_API_KEY` (+ gateway `ANTHROPIC_BASE_URL`) |
| GPT-family | `codex` | |
| Qwen-family | `qwen-coder` | |
| Grok-family | `grok-build` | |
| Gemini-family | `gemini-cli` or via the claude-code gateway | pick by the protocols your gateway supports |

Evaluation convention: always request the highest reasoning tier
(`--agent-kwarg reasoning_effort=max`); within one round, use the same agent
configuration for all trials so results are comparable across models.
