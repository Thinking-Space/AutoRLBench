# Qwen3-VL fused sequence-parallel label misalignment

Submitter: Gaoyuan. Created: 2026-10-24. Domain: code; framework: verl; algorithm: GRPO. Sources: https://github.com/verl-project/verl/issues/6068 and https://github.com/verl-project/verl/pull/6268. Base commit4b9c14f2c306d43f84eccf47315323e3f1ce1270; paired repair consists of six production-file changes from the linked PR plus identical public compatibility changes. Source/patch/resource SHA and licenses are in evaluations/issue/reference/provenance.

One issue-level diagnostic task, one public recorded session. The complete source workspace preserves necessary upstream modules/tests and the recorded compatibility changes. Detached answer-bearing observers and unrelated agent instructions are moved out of the public source; the transformation inventory and arithmetic-AST comparison are hidden references. The target reference patch applies to the delivered source.

Recipe: Qwen3-VL-8B-Instruct, ScienceQA image train2560/validation1024, seed42, FSDP4 FULL_SHARD DP2×SP2, fused/remove-padding, batch128/mini32/n8/micro1, lr1e-6, lengths16384+4096, scheduler200 and normal stop40. Two independent original-weight formal runs, one per arm. Each has40 updates,160 AdamW calls per rank,9 validations,scalar/W&B0–40 and step20/40 nonempty checkpoint inventories. Sampling n1 is the recorded validation protocol.

|Metric|Before|After|
|---|---:|---:|
|First20 reward|0.951416015625|0.967578125|
|Last20 reward|0.91787109375|0.982861328125|
|Initial validation|0.9384765625|0.9462890625|
|Step40 validation|0.900390625|0.97265625|
|Last5 validation change|−2.578125pp|+2.7734375pp|

Paired validation-change improvement=+5.3515625pp. Frozen before quality-drop criterion (≤−5pp): false. Response-label / nonzero-advantage / wrong-position-gradient observations: before16,311,325/2,896,350/381,610; after0/0/0. Counts are repeated capped events. Experiment scope is one seed and a public replacement for the source issue's internal dataset.

Historical allocation ledger:59.2506188121107 GPU card-hours across17 closed sessions. Before used a historical finite deadline that did not trigger; after usedinf and an unbounded parent wait. Both normal stop40. Large model/data resources use pinned manifests and online/cache preparation; checkpoints and full tensor observations stay at the recorded remote experiment. Tensor SHA inventories are included; checkpoint shard SHA and restore test status arefalse.

Environment integration: task.toml and AUTORLBENCH_* channels follow the current AutoRLBench interface; the recorded training provenance retains its original revisions. It adds a registered image definition and its actual dependency lock while preserving the official loader/validator. The image uses a digest-pinned base and an isolated torch2.8/cu129, vLLM0.11/cu129, Python3.12 runtime. Build/clean prepare/training status is recorded separately from schema validation.

Issue verifier: the current framework's shared OpenAI-compatible judge is reused unchanged. It reads agent_answer.md and hidden oracle.md, requires AUTORLBENCH_JUDGE_MODEL,OPENAI_BASE_URL,OPENAI_API_KEY, and writes result.json. The judge was not invoked during construction. No solution-level evaluation is declared. The hidden six-file patch and successful paired mechanism metrics are reference materials.

Only evaluations/issue/instruction.md and workspace belong in an agent sandbox. Metadata, oracle, paired references and build/audit material stay outside that sandbox. Public log paths were mapped to/workspace; numerical values and record order are preserved. Build audit records state the implemented visibility boundary and static checks.
