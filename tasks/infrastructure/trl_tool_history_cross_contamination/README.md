# Python repair continuations use another rollout's history

Qwen3-4B is trained with GRPO on real Python repairs; the vLLM server continues 338 of 844 post-tool samples from a different history than the receiving rollout.

| | |
| --- | --- |
| **Author** | PurvaChiniya |
| **Domain** | infrastructure |
| **Tags** | `trl` `tool-use` `issue-only` |
| **Model** | Qwen/Qwen3-4B |
| **Dataset** | BugsInPy: 16 train, 4 development, 8 sealed bugs; two projects |
| **Framework** | TRL, `run.sh` |
| **Environment** | `tool-rl-vllm024`; recorded on 2 × H100 80 GB, 24 CPUs, 128 GiB RAM |
| **Evaluations** | issue |
| **Recorded runs** | 1 defective + 1 control, seed 17, 3 optimizer updates; 488 s + 364 s including startup and control replay |

## Difficulty

`GRPOTrainer._generate_single_turn` reuses four generations after tool use. `VLLMGeneration.generate` consequently takes every fourth input history and draws four outputs from each retained history. The optimization is valid for initial shared prompts but invalid after independently executed tools. Shapes remain valid and the model sometimes repairs bugs: development success is 4/16 trajectories, or 1/4 bugs, at steps 0–3 in both arms. The short run does **not** establish a repair-success improvement or generalization gain.

Across training and development evaluation, the defective run has 338/844 continuation mismatches, versus 0/832 after the upstream fix. Initial inputs have 0/160 mismatches in each arm. Among rows in naturally divergent continuation groups, the counts are 337/615 versus 0/357. The actual server's echoed prompt token IDs match the submitted client IDs throughout. Replaying captured natural histories through both production dispatch functions gives 1/4 versus 0/4 mismatches; identical-history and initial-generation controls both give 0/4. Full request and tool evidence is retained because correspondence, rather than reward alone, establishes the failure.

## Reference solution

**Issue:** identify incorrect continuation multiplicity and resulting prompt subsampling, distinguish initial shared prompts from later independent histories, and connect raw request tokens to the receiving trajectory. Low coding accuracy, stale weights, tool parsing, workspace races, and insufficient context alone are inadequate explanations. The reference repair requests one continuation per post-tool history while retaining four initial samples and server mode; see [repair.patch](reference/repair.patch).

The cases are original BugsInPy defects, with unchanged regression/non-regression tests from pinned reference revisions. All 28 defective cases fail and all 28 reference repairs pass on Linux. Related source files share a split. Eight sealed cases were admission-tested on CPU but never evaluated with the policy. Tool calls are capped at six; tests run on fresh copies with fixed commands, collection-count checks, and resource limits. This controlled tool interface is not an OS security sandbox. Public-case familiarity and the small development set limit conclusions.

## Verification

**Issue:** the shared LLM judge compares `agent_answer.md` with the hidden oracle, weighting verdict/root cause/evidence 0.2/0.6/0.2; passing requires at least 0.8. No solution tier or performance threshold is claimed: both short arms remain at the repaired step-zero development score. The proposed three-seed, 60-update campaign was not run.

CPU diagnostics execute the exact upstream functions; the real-server pilot and replay separately confirm the production defect. Raw W&B binaries are retained under [reference/raw_wandb](reference/raw_wandb); all per-step metrics and representative full rollouts remain in the historical workspace. GPU runs used the pinned Modal runtime. See [verification.md](reference/verification.md) for benchmark Dockerfile, preparation, and structural checks. No judge-model evaluation has been run.
