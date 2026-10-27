# Third-party notices

This notice describes the local bundle; it is not a claim that every component
has been cleared for public redistribution. Preserve upstream notices and
file-level licenses. The root MIT grant covers original task contributions only.

| Material | Source / attribution | Applicable terms and scope |
| --- | --- | --- |
| `workspace/idea_source/` | IDEA project implementation; retained upstream framework attribution: SEED / AIMING Lab, copyright 2026; additional attributions in `Notice.txt` | The bundled `workspace/idea_source/LICENSE` declares MIT. Preserve `Notice.txt`, file headers, and any more-specific vendored licenses. The original upstream README is retained in `evaluations/issue/verifier/resources/author_materials/upstream_notices/framework-README.md`; project renaming does not alter copyright or citations. |
| `workspace/awm_source/` | Supplied from local `agent-world-model`; upstream Snowflake-Labs/agent-world-model; captured commit `85e322f69279e3b3325b7377ec3bab788514e9cb` plus recorded worktree changes | MIT License added by upstream commit `28342e7c7f23ce93e8c8b14b881c9c4adb885d8b`; the complete upstream text is retained as `workspace/awm_source/LICENSE`. The task does not relicense independent dependencies or data. |
| Removed benchmark integrations | The upstream `mcp-adapted-bench` submodule and AWM benchmark entry points | Removed from the task because they are unrelated to the RL diagnosis. They are not part of the distributed workspace. |
| Prepared `workspace/data/AgentWorldModel-1K/` | Snowflake/AgentWorldModel-1K; Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He; downloaded from the pinned `locaszzz/AgentWorldModel-1K` snapshot with its data card | The data card declares CC BY 4.0. Preserve attribution, the data card and its citation. Applicable license: https://creativecommons.org/licenses/by/4.0/ . Runtime code embedded in data is not covered by the root MIT grant. |
| Task train/test selection | `locaszzz/IDEA_training_data`, derived from AgentWorldModel-1K | Upstream data terms and attribution remain applicable. Inputs are directly readable original files checked by `environment/data_manifest.json`. |
| `evaluations/issue/instruction.md`, `evaluations/issue/verifier/{test.sh,judge.py,judge_prompt.md}` | AutoRLBench; copied from its reward-task evaluation scaffolding | Retain applicable Apache-2.0 upstream terms. The upstream Apache-2.0 license text is included as `LICENSES/Apache-2.0.txt`. |
| `workspace/logs/` and reviewer evidence | Historical training output contributed by locaszzz, with excerpts of upstream tasks, generated environments and verifier outputs | Original annotations follow the root MIT grant; embedded upstream content retains its applicable terms. Sanitization and curated selection do not change numerical rewards or create new rights in embedded material. |
| `locaszzz/IDEA_sft` | Model initialization referenced by this task | Model weights are not shipped. Model access and use remain subject to the model repository's own terms; this bundle grants no model-weight license. |

## Data attribution and packaging changes

Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement
Learning (2026), by Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han,
Zhewei Yao, Huaxiu Yao, and Yuxiong He. The retained data card identifies
Snowflake/AgentWorldModel-1K and the upstream paper, arXiv:2602.10090.

The five generated JSONL inputs and train/test parquet files used by the
recorded code-reward path are downloaded during prepare from
`locaszzz/AgentWorldModel-1K` at commit
`6f8a98a043c631ba9c88dd1d1d1d1829a1f9dca5`, including the upstream data card.
Their original hashes and sizes are in `environment/data_manifest.json`; the
dataset files are not included in the submission tree. Unused synthesis
inputs and SQL/LLM-mode verifier data are omitted. Packaging relocates paths,
redacts credentials/host identifiers, and retains 20 complete historical episodes,
including 8 audited reward counterexamples. Source changes and evidence are
indexed by `environment/source_manifest.json` and the hidden
`evaluations/issue/verifier/resources/curated_trajectory_manifest.json`.
This does not imply endorsement by
upstream authors, a training rerun, or permission to redistribute an otherwise
unlicensed component.

Unrelated non-AWM environment implementations, site documentation, image assets
and alternative source-tree Dockerfiles are omitted from this task-specific
release. The retained server logs are directly readable text files. These scope
changes do not repair reward rules, alter historical numerical evidence or change
upstream authorship and the terms applying to retained material.
