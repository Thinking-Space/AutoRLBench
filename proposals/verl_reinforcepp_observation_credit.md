# Proposal: diagnose broken credit assignment in an RL-trained SQL analyst

**Status:** design proposal. The upstream defect and repair are documented; this workload, its training runs, difficulty, and performance gap have not been validated.

| Field | Proposed value |
| --- | --- |
| Task ID | `verl_reinforcepp_observation_credit` |
| Domain | algorithm |
| Workload | Multi-step operations analytics over a local relational database |
| Policy / algorithm | Qwen3-4B, non-thinking mode, REINFORCE++ |
| Evaluation | Issue diagnosis; solution conditional on validated repaired runs |

This proposes a reproducible failed RL run for AutoRLBench's debugging agent. A SQL policy may produce valid queries and plausible answers while failing to improve earlier analytical decisions; that behavior is a hypothesis to test. The established framework defect clears the running return at tool observations, cutting terminal-reward propagation to earlier actions. The benchmark challenge is to establish that causal chain from code, trajectories, and training evidence.

## Practical workload

Train an analyst over a small SQLite warehouse containing orders, order items, refunds, shipments, and shipment events. Use generated, internally consistent business records. Publish the database generator and reference queries; these are synthetic operations data, not customer data or production traces.

Limit the initial scope to three query families: refund-adjusted revenue, late-delivery rates, and partial fulfillment. An example is: “Which warehouses' late-delivery rates increased last month, excluding cancelled orders and counting split shipments only once per order?” The policy discovers the schema, inspects relevant records or intermediate aggregates, and submits executable SQL plus its answer. Exclude customer cohorts and broad warehouse exploration from the first version.

Provide schema and read-only SQL tools. Target about three meaningful tool interactions and cap episodes at five calls. Do not enforce a minimum or reward additional calls; report natural interaction counts. Legitimate nulls, one-to-many joins, repeated events, and bounded SQL errors provide practical reasoning demands. The pilot must show that earlier decisions genuinely use intermediate observations and that the 4B policy sometimes succeeds.

The terminal reward comes from a protected query evaluator. Compare the submitted query with a reference on the episode database and withheld perturbations to reject hardcoded answers and accidental agreement. Check the submitted answer on the episode database. Preserve duplicate semantics, normalize ordering only when permitted, and specify numeric tolerances. Do not add intermediate-answer, length, or tool-count rewards.

## Diagnostic value

The defective run may retain nonzero gradients, valid SQL, and improving final-answer formatting. These are hypotheses to measure, not observed results. The agent must distinguish incorrect credit assignment from difficult joins, sparse rewards, tool-output truncation, context limits, or a faulty scorer.

Include single-query controls and questions whose early choices affect outcomes several observations later. Report performance by dependency depth. The diagnostic question is whether failure follows observation boundaries after accounting for difficulty and context limits.

Provide normal training code, configuration, metrics, logs, and complete sampled trajectories. Initial instructions describe symptoms without naming the offending function or linking the upstream fix. The reference patch, paired analysis, and author-generated causal annotations remain in reviewer/verifier materials; raw evidence needed to derive the diagnosis stays available.

## Established upstream mechanism

- [Issue #7278](https://github.com/verl-project/verl/issues/7278)
- [Merged repair #7300](https://github.com/verl-project/verl/pull/7300)
- Defective parent: `f773c96a7f5fc5e28697273160a6dcf5fa99864d`
- Repaired commit: `8a1bf6d5b080173e29834aca55ee8d47549b2ea6`
- Function: `compute_reinforce_plus_plus_outcome_advantage`, `verl/trainer/ppo/core_algos.py`

The response mask excludes observations and padding. The defective reverse scan multiplies its running return by this mask, clearing it at internal observations. The upstream CPU example changes valid-token returns from `[1, 1, 1, 1]` to `[0, 0, 1, 1]` when an observation span is inserted. The repair carries return through observations while excluding those positions from the objective. Zero raw returns do not imply zero whitened advantages or gradients.

We propose a new recorded workload reproducing a known bug; we do not claim discovery or an already measured learning improvement.

## Experiment and evidence

Use one pinned runtime and change only the upstream production-code patch between arms. Apply identical, disclosed compatibility adjustments if needed. Target full parameter training with gradient checkpointing and a memory-managed colocated veRL rollout engine on two GPUs; validate this execution layout in the pilot.

Provisional recipe: 16 prompts per update, four trajectories per prompt, 8,192 total context tokens, at most 2,048 generated assistant tokens across an episode, and 100 optimizer updates. Freeze the recipe after the pilot and before formal runs; report actual tool-turn counts and truncation.

Target 512 training questions, 96 development questions, and 128 sealed evaluation questions. Split by reference-query template and database seed, keeping paraphrases and equivalent templates together. The three broad families may occur in multiple splits; reserve specified combinations of joins and filters for a separately reported held-out composition stratum. Publish generator seeds, manifests, hashes, and reference-query checks. Freeze splits before training and use no sealed evaluation results for recipe or threshold tuning.

Run three matched seeds per arm: six formal runs. Compare the fixed-budget endpoint and identical evaluation schedules, including step zero. Report every seed and paired changes. Three seeds do not establish broad variance or cross-hardware guarantees.

Required controls and evidence:

1. Replay captured successful and failed trajectories through both return implementations using identical tokens, rewards, and masks. Verify early-turn credit changes.
2. Include single-turn and observation-free compacted traces as negative controls; test gamma below one and trailing padding.
3. Verify observations remain excluded from policy loss. Unmasking tool output is not a valid repair.
4. Recompute rewards with the protected SQL evaluator. Reference queries must pass and deliberately incorrect joins must fail.
5. Record per-turn returns/advantages, entropy, gradient norm, KL if configured, context truncation, valid-SQL rate, task success, and dependency depth. Check that parameter updates reach the rollout policy.

## Benchmark evaluation

**Issue:** identify the observation-boundary reset, distinguish observation from termination, and connect it to measured credit and training symptoms. Address the strongest plausible alternative supported by the evidence. Use the benchmark's shared issue-judge protocol.

**Solution:** the agent repairs the workspace and submits `report.md`; the harness reruns the frozen experiment. Preserve the workload, algorithm, terminal reward, sampling multiplicity, tool limits, and training budget. Switching algorithms, removing tools, unmasking observations, changing the SQL scorer, or introducing intermediate rewards does not satisfy the contract.

Require complete fresh training, protected reward re-evaluation, correct credit propagation/masking, and a performance criterion supported by development reference runs. Calibrate a numerical threshold only after observing development separation, with a margin over the repaired step-zero baseline. Freeze the gate before reporting sealed-test results. Do not select a threshold from those sealed results.

If the mechanism reproduces but learning does not separate, propose issue-only with bounded claims. Seek maintainer agreement before using a mechanism-only solution gate.

## Staged reproduction and compute

First reproduce the upstream return discrepancy on CPU, then exercise the training path with a small lookup smoke workload. That smoke test is a diagnostic aid, not a separate proposed benchmark task. Validate the SQL generator and evaluator before GPU experiments. Run a short paired SQL pilot before the six formal runs.

All figures below are estimates for the proposed 4B workload, not measured runtimes. GPU-hours count every allocated GPU, including time spent waiting on tools.

| Stage | Allocation / budget |
| --- | --- |
| CPU preparation | Database/reference validation, upstream diagnostic, runtime pinning |
| GPU pilot | 2 × H100 80 GB for 2–3 elapsed hours: 4–6 GPU-hours |
| Each formal run | Approximately 2–4 elapsed hours on two H100s |
| Six formal runs | Approximately 24–48 GPU-hours |
| Complete contribution | Reserve 40–80 GPU-hours, including controls and verifier reruns |
| Host | 24–32 CPU cores, 128 GB RAM, 250 GB free SSD |
| Engineering | Approximately 3–5 working days |

A100 80 GB cards are a possible alternative, but time must be re-estimated from measured throughput. A 1.7B smoke test cannot substitute for the frozen 4B comparison.

Pilot exit criteria: reliable reference evaluation, genuine model successes below ceiling, meaningful internal observations, controlled truncation, and the defect visible in the real training path. Measure peak memory, update time, and tool latency before committing the full budget. Revise or stop the workload if these conditions fail.

## Scope and requested feedback

Public overlap check: 27 October 2026, AutoRLBench main `47b23eb1aec3b5d70e842229a4978790231b8061` and open PRs #1–#9. No matching case was found. Existing GRPO grouping and policy-ratio cases have different mechanisms. Private reservations are unknown.

This PR contains the design only, under `proposals/`; that is a suggested discussion location rather than an existing task convention. No runnable task, recorded training evidence, or validator pass is claimed. The contribution guide explicitly allows ideas for constructing a workspace.

Maintainer feedback requested: is this mechanism unreserved, does the bounded synthetic SQL workload fit the benchmark, and is the issue-first evaluation with a conditional solution level appropriate?

The final task will include defective-run evidence, a minimal hidden repair, paired control runs, a digest-pinned environment, asset manifests, and evaluation scaffolding. Preserve full stdout/stderr, all per-step metrics, offline W&B including its binary in the raw bundle, and sampled complete trajectories. Package and sanitize using the repository workflow. Report structural checks, diagnostic replays, and fresh GPU validation separately.
