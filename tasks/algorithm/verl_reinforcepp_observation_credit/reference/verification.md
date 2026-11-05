# Verification record

Recorded on November 3, 2026 PT (November 4 UTC), in the authorized `scale-rsi` Modal environment `improve-vlm-gui-grounding`, app `pc_autorlbench`.

## SQL evidence

- [Defective GPU run](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-iLZDyW7GFW40xDqWT9kIop): completed, 388.615 s, seed 17, three updates.
- [Repaired GPU run](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-REcRSgJdVhdq0DpSkGsCZz): completed, 389.999 s, same budget and seed.
- Each run used 2 × H100 80 GB, 24 CPUs, 128 GiB RAM. The completed pair consumed about 0.433 H100 GPU-hours of function runtime; this excludes earlier pilots, startup billing differences, image builds, storage, and CPU audits. It is not a billing statement.
- Defective: 192 traced trajectories, four rewarded tool trajectories, all 2,480 earlier action tokens have raw return zero. Control: one rewarded tool trajectory, all 587 earlier action tokens have raw return one. Development accuracy is zero at steps 0–3 in both arms.
- `credit_replay.json` replays identical full batches through both exact upstream functions: the defective outputs match the recorded returns and the repaired scan restores earlier reward. This is stronger causal evidence than comparing independently sampled trajectories.
- Exact-function CPU diagnostics cover an internal observation gap, an observation-free control, discounting, and trailing padding. See `cpu_diagnostic.json`.
- All 28 packaged SQL workload tests pass, including reference-query correctness on perturbed databases, rejection of hardcoded answers and fanout errors, query limits, and mutation rejection.

Upstream provenance: [issue #7278](https://github.com/volcengine/verl/issues/7278), [repair #7300](https://github.com/volcengine/verl/pull/7300). Parent `f773c96a7f5fc5e28697273160a6dcf5fa99864d`; repair `8a1bf6d5b080173e29834aca55ee8d47549b2ea6`. The common runtime adaptation is preserved in `runtime-compatibility.patch`.

## Reproduce and inspect

After the benchmark preparation stage, run `workspace/run.sh` on two H100-class 80 GB GPUs to repeat the three-update recorded recipe (roughly 6.5 minutes plus preparation). Use a fresh `LOG_DIR` for each run. For a control, apply `reference/repair.patch` inside a separate copy of `workspace/vendor/verl`, preserving the workload and runtime. Asynchronous sampling prevents bitwise reproducibility across independent runs.

Run `python reference/analyze.py sql workspace/observed_run` to reconstruct the counts from raw traces. Run `python -m pytest -q reference/tests` from this task directory for the 28 CPU workload checks. Hidden controls and patches are reviewer materials, not agent-visible grading flags.

## Runtime and package checks

The actual `docker/images/tool-rl-vllm024.Dockerfile` was built successfully with Modal's Dockerfile builder. Both final `prepare.sh` scripts completed on that image: all model files passed their revision/SHA-256/size checks, TRL corpus source/test downloads passed their checks, the SQL generator ran, and both frameworks imported. The corpus builder also passed a second, idempotent invocation. A prepared original regression case failed with the expected test counts. These were CPU checks, not a new GPU training campaign or an end-to-end Harbor judge run.

- [Final CPU preparation check](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-Suv4HUVqibARLXn4qWMbd3)
- Python 3.12.10, torch 2.11.0, vLLM 0.24.0, transformers 5.10.1, datasets 5.1.0, accelerate 1.15.0, wandb 0.31.0. Legacy repair tests use Python 3.9.25.
- Base-image digest is pinned; the new image's tooling block exactly matches the canonical `trl-grpo-torch26` block. Each asset has a pinned revision, hash, and byte size.
- `python -m autorlbench validate <task>`: zero errors. Canonical issue instruction and verifier files compare byte-for-byte with the benchmark template.
- Source-manifest checks cover every packaged workspace file. Edits to historical artifacts affect labels/paths only; metric values and raw token IDs remain unchanged. No model weights, checkpoints, credentials, or raw W&B binaries are included in the agent workspace.

Repository checks: 64 passed and 3 failed across environment-plan, training-script, and SQL workload tests. All three failures reproduce when the new image is excluded: existing images have an unpinned base, an underscore-containing image tag, and divergent tooling blocks. These pre-existing failures were not changed. The added image passes the corresponding targeted checks.

## Complete debugging-agent trial

On November 4, 2026, one fresh Harbor Terminus-2 agent (GPT-5.4) inspected the published task at commit `4039e6059bfa634032c33c4d7c62285a3229dcdf`. The canonical issue verifier then graded its answer using GPT-5.2. **The workflow completed without a Harbor trial error; the agent failed with score 0.2/1.0.** Verdict score was 1.0; root-cause and evidence scores were both 0.0. The agent blamed incorrect SQL joins instead of identifying the return accumulator reset across tool observations.

The run used a CPU-only Modal sandbox (4 CPUs, 16 GiB), with a separate 2-CPU/8-GiB controller. Agent execution took 65.016 seconds; controller runtime including setup and grading was 279.237 seconds. The agent stopped after eight episodes within its 24-turn/900-second limit. There was one actual agent trial and no retries. Two setup errors were corrected before agent execution. The command trace includes large file dumps and interrupted heredocs; these are part of the recorded attempt.

Instruction, hidden oracle, and verifier files matched the evaluated commit byte-for-byte. The agent used a dedicated prepared workspace volume with network access disabled; reference controls and the oracle were outside its input workspace. The verifier received its hidden files after agent execution. A post-run SHA-256 comparison found all 504 original non-model files unchanged. Model assets were checked during preparation and excluded from this post-run audit.

See [agent-evaluation-01](agent-evaluation-01/README.md) for the exact answer, canonical judge output, command trace, settings, provenance, and limitations. This records one failed diagnosis; it is not a pass-rate estimate.

## Scope

This is an **issue-only** contribution. One paired seed and three optimizer updates per arm verify a production mechanism. They do not establish a learning-performance improvement, calibrated solution threshold, multi-seed reliability, or sealed-test generalization. One complete agent-and-judge trial has now run, with a failed diagnosis as recorded above. Full W&B binaries are retained under `reference/raw_wandb`, outside the historical agent workspace. Modal logs are supplementary; the committed traces and metric files contain the evidence needed for diagnosis.
