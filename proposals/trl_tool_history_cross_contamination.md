# Proposal: diagnose TRL tool-history corruption in small Python repairs

**Status:** design proposal. The upstream bug and repair are documented; the corpus subset, real-server training runs, difficulty, and performance gap have not been validated.

| Field | Proposed value |
| --- | --- |
| Task ID | `trl_tool_history_cross_contamination` |
| Domain | infrastructure; subject to maintainer categorization |
| Workload | Short, single-file Python repairs in isolated repository workspaces |
| Policy / algorithm | Qwen3-4B, non-thinking mode, GRPO with four samples per prompt |
| Evaluation | Issue diagnosis; solution conditional on validated reference runs |

The task asks a debugging agent to explain poor repair learning caused by a broken association between a rollout and its history. Policy samples start with the same bug report, then inspect and edit their own workspaces differently. The defective server path generates later actions from another sample's history. A plausible edit can therefore refer to test results or changes the receiving sample never observed.

## Practical workload

Curate small real Python defects, starting with [BugsInPy](https://github.com/soarsmu/BugsInPy) and its defective/fixed revisions. Target two or three compatible pure-Python codebases, single-file fixes of roughly 5–20 changed lines, and focused tests that complete within about five seconds on the reference host. These are selection targets, not verified corpus properties. Exclude compiled extensions, external services, network access during episodes, and cases that require extensive repository exploration. Pin the interpreter and dependencies, audit builds before GPU work, and preserve source licenses and provenance. Exact cases have not yet been selected.

Each episode provides a bug description, failing-test summary, localized repository snapshot, and likely module path without revealing the fix. Tools support file read/search, patch application, and a fixed test command. Target three to five calls for inspect–patch–test with an optional revision; cap at six. Do not require unnecessary calls. A successful patch must pass protected regression and non-regression tests on a fresh checkout.

All four samples of one prompt start from byte-identical repositories, tests, and environment state, with independent workspaces. Histories diverge through the model's own choices. Do not manufacture divergence with different hidden tasks, random tool outputs, per-sample secrets, or nonce-copying objectives.

Reward program behavior. A protected evaluator applies allowed source changes to a clean checkout and uses its own unchanged test definitions and runtime configuration. Invalid patches, resource-limit violations, empty test collection, edited tests, or disabled assertions cannot earn success. The source project's defect is the coding workload; the AutoRLBench root cause is the training framework's dispatch.

## Diagnostic value

A continuation from another branch can be syntactically valid and refer to the same repository. Symptoms may resemble weak coding ability: irrelevant edits, repeated tests, or apparent disregard for recent feedback. Training can retain nonzero rewards and losses. These are expected failure modes to investigate, not observed results.

The agent must distinguish misrouted history from insufficient context, flaky tests, failed edits, stale policy weights, and shared-workspace races. Evidence must connect actual server input tokens to subsequent actions rather than merely show low repair success.

Supply normal request/group/trajectory IDs, tool logs, model requests, workspace/patch hashes, and training metrics. Keep upstream issue links, the reference patch, derived mismatch labels, and paired causal analysis in reviewer/verifier materials. Retain the raw evidence needed to reconstruct correspondence.

## Established upstream mechanism

- [Issue #7416](https://github.com/huggingface/trl/issues/7416)
- [Merged repair #7418](https://github.com/huggingface/trl/pull/7418)
- Defective parent: `4c2d04825b7129140b8e8ed07c8399debe06f7bd`
- Repaired commit: `d947c4f5098c8d6ca30ea7ff58b98e6570fd6a01`
- Affected path: GRPO tool continuation through vLLM server mode.

Initial generation legitimately draws four samples from a shared prompt. After tool use, histories differ. The defective continuation path retains the initial sampling multiplicity, so prompt subsampling keeps every Nth history and generates N continuations from it. The repair requests one continuation per sample at that stage while preserving four initial samples.

Upstream CPU and trainer tests establish the dispatch error; they do not establish real-server learning improvement in this coding workload. We do not claim discovery of the bug.

## Experiment and evidence

Use one pinned TRL runtime with an actual vLLM server and isolated CPU coding environments. Change only the upstream production-code repair between arms. Compatibility adjustments must be identical and disclosed. Preserve server mode and four initial samples.

Target full parameter training with gradient checkpointing, one GPU for the 4B trainer and one for inference. Pilot recipe: eight prompts per generation batch, four samples each, 4,096 total context tokens, at most 1,024 generated assistant tokens per episode, and 60 optimizer updates. Start with microbatch one and accumulation as required. Keep the 4B policy to reduce the risk that coding capability dominates the result; savings come from shorter episodes and a smaller workload. Freeze model revision, recipe, test commands, and limits before formal runs.

Initial corpus target: 16 training defects, four development defects, and eight sealed evaluation defects from the prevalidated codebases. Publish exact IDs, upstream revisions, regression tests, and split manifests before formal runs. Keep related fixes, duplicates, and aliases in one split. Describe this as held-out bugs, not held-out projects unless that is actually achieved. Report counts as well as percentages; eight evaluation bugs give coarse estimates, and repeated samples or training seeds do not increase the number of independent bugs. This is a focused framework-failure study, not a coding leaderboard. Public provenance means pretrained-model familiarity cannot be ruled out.

Run three matched seeds per arm: six formal runs. Report fixed-budget endpoints, step-zero results, natural history-divergence rates, and paired seed-level differences. Preserve the same sampling multiplicity during evaluation and report both per-trajectory repair success and the number of distinct bugs repaired. Do not select the best checkpoint or seed after examining sealed results.

Required controls and evidence:

1. Record each sample's expected serialized history and actual server-side input token IDs, with row order, group/request IDs, and workspace identity. Compare exact tokens after accounting for documented serialization.
2. Report mismatch counts and denominators for all continuations, distinct-history continuations, and affected groups. Initial duplicate prompts are legitimate.
3. Replay naturally divergent captured histories through both dispatch paths with an actual server. A CPU mock alone is insufficient.
4. Include identical-history and single-turn controls. One-sample diagnosis may isolate the trigger but is not an allowed benchmark repair.
5. Verify filesystem isolation and clean resets. Independently apply patches and run protected tests; record timeout, flakiness, and test-collection checks.
6. Check policy-to-server weight updates, parse failures, context truncation, rollout counts, patch applicability, and repair success to evaluate alternative explanations.

## Benchmark evaluation

**Issue:** identify incorrect continuation multiplicity and prompt subsampling after tool use; distinguish initial shared prompts from later distinct histories; connect request/history mismatches to recorded failures. Generic claims of context loss or weak tool use are insufficient. Use the benchmark's shared issue-judge protocol.

**Solution:** the agent repairs the workspace and submits `report.md`; the harness launches the frozen real-server experiment. Preserve four initial samples, isolated workspaces, server mode, tool interface, evaluator, and training budget. Reducing initial samples to one, disabling tools, changing to colocate mode, or sharing workspaces does not satisfy the contract.

The verifier must independently establish complete fresh training, server/history correspondence, preserved initial group size, and patch outcomes on clean workspaces using protected tests. Historical logs and agent-written success flags cannot provide a pass. Grading instrumentation belongs to the harness/verifier and cannot be replaced by the submitted workspace.

Issue diagnosis is the initial submission target. A real failed training run with traced history corruption is useful evidence even if a short training budget does not produce a reliable repair-success gap.

Add solution evaluation only if reference runs support a defensible gate. Set any repair-success threshold from development reference runs, requiring separation and improvement over the repaired step-zero baseline; freeze it before sealed evaluation. With only four development and eight evaluation bugs, a stable threshold may be impossible. In that case retain issue-only rather than promise performance grading. A mechanism-only solution gate would require explicit maintainer agreement.

## Staged reproduction and compute

Begin with the upstream CPU diagnostic and a small lookup/tool smoke test, then verify naturally divergent code-repair histories through an actual server. The lookup test is a development aid, not a separate submission. Build and validate the selected coding environments before GPU work, then run a short paired pilot before formal runs.

All figures below are estimates for the scoped 4B workload, not measured runtimes. GPU-hours include both the trainer and server GPU, including CPU tool waits.

| Stage | Allocation / budget |
| --- | --- |
| CPU preparation | Corpus builds, protected tests, isolation/reset validation, upstream diagnostic |
| GPU pilot | 2 × H100 80 GB for 2–3 elapsed hours: 4–6 GPU-hours |
| Each formal run | Approximately 1–2 elapsed hours on two H100s, including CPU tool waits |
| Six formal runs | Approximately 12–24 GPU-hours |
| Complete contribution | Reserve 24–48 GPU-hours, including controls and any verifier reruns |
| Host | 16–24 CPU cores, 128 GB RAM, 200 GB free SSD; bounded sandbox concurrency |
| Engineering | Approximately 3–5 working days; corpus selection and compatibility remain schedule risks |

The layout follows [TRL server-mode guidance](https://huggingface.co/docs/trl/vllm_integration). A100 80 GB cards may work, but timings require a new throughput estimate. Memory fit for full parameter training is a pilot requirement, not a measured claim.

The pilot must demonstrate genuine repair successes below ceiling, stable tests, naturally divergent histories, and the production defect. Measure peak memory, update time, sandbox startup, test latency, and truncation. Check correspondence on actual server inputs, not only a mock. If the 4B model never repairs a case, histories rarely diverge, or the corpus cannot meet the short-test constraint, revise the workload before formal runs. Do not manufacture divergence or add unrelated bugs to obtain a gap.

## Scope and requested feedback

Public overlap check: 27 October 2026, AutoRLBench main `47b23eb1aec3b5d70e842229a4978790231b8061` and open PRs #1–#9. No matching case was found. This differs from identical sampling seeds (PR #6), fused sequence-parallel label alignment (PR #7), and GRPO UID grouping. Private reservations are unknown.

This PR contains the design only, under `proposals/`; that is a suggested discussion location rather than an existing task convention. No selected corpus, runnable task, recorded training outcome, or validator pass is claimed. The contribution guide explicitly allows ideas for constructing a workspace.

Maintainer feedback requested: is this mechanism unreserved, does a small TRL code-repair workload fit the benchmark, and is infrastructure the appropriate domain? We propose issue diagnosis first and a solution level only after validated reference runs. Please advise if a protected fresh-run correspondence gate would be acceptable when repair-success separation is weak.

The final task will include a pinned environment, source/corpus manifests, defective-run evidence, hidden paired controls and reference repair, and evaluation scaffolding. Preserve stdout/stderr, per-step metrics, raw offline W&B including its binary, and representative full trajectories. Provide model asset retrieval instructions and report structural checks, diagnostic replays, and fresh GPU validation separately.
