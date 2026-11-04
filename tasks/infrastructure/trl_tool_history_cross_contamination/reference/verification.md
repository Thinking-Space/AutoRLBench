# Verification record

Recorded on November 3, 2026 PT (November 4 UTC), in the authorized `scale-rsi` Modal environment `improve-vlm-gui-grounding`, app `pc_autorlbench`.

## Python-repair evidence

- [Defective GPU run](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-BBShXz6ouEOzPcicgXbFFD): completed, 487.953 s, seed 17, three updates.
- [Repaired GPU run](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-5LTxMpaZ6rrCK6qXjE2HLt): completed, 364.153 s including post-training server replay, same training budget and seed.
- Each run used 2 × H100 80 GB, 24 CPUs, 128 GiB RAM. The completed pair consumed about 0.473 H100 GPU-hours of function runtime; this excludes earlier attempts, startup billing differences, image builds, storage, and CPU audits. It is not a billing statement.
- Real server correspondence: 338/844 continuation mismatches before repair; 0/832 after. Both initial-generation controls have 0/160. The server's echoed token IDs match client submissions in both arms. Naturally divergent groups have 337/615 versus 0/357 mismatches.
- Real-server replay of captured histories: 1/4 versus 0/4 mismatches for natural divergence; 0/4 in both identical-history and initial-generation controls. See `control/dispatch_replay.json`. The separate CPU diagnostic uses mocked transport and is labeled accordingly.
- Development success is 4/16 trajectories and 1/4 bugs at steps 0–3 in both arms. Nonzero training rewards and gradients demonstrate that this was an executed optimization run, but no accuracy benefit was established.
- All 28 original corpus cases fail and all 28 reference patches pass their full focused test modules on Linux. Five packaged integration tests pass locally: paired admission, independent workspaces with identical resets, protected grading tests, tool limits, and source-group split separation.

Upstream provenance: [issue #7416](https://github.com/huggingface/trl/issues/7416), [repair #7418](https://github.com/huggingface/trl/pull/7418). Parent `4c2d04825b7129140b8e8ed07c8399debe06f7bd`; repair `d947c4f5098c8d6ca30ea7ff58b98e6570fd6a01`. BugsInPy revision `11c5f1eea954a42132cfd06bf257766a7963e0fd`; exact cases and source/test revisions are in the data manifest. `corpus_admission.json` and reviewer-only `corpus_patches/` preserve admission evidence and reference fixes. Preparation downloads original production source and reference tests, not repaired production source.

## Reproduce and inspect

After the benchmark preparation stage, run `workspace/run.sh` on two H100-class 80 GB GPUs to repeat the three-update recipe (roughly 6–9 minutes plus preparation). One GPU trains; one runs the real vLLM server. Use a fresh `LOG_DIR` for every run. For a control, apply `reference/repair.patch` inside a separate copy of `workspace/vendor/trl`, retaining four initial generations and the same server mode. Asynchronous sampling prevents bitwise reproducibility across independent runs.

Run `python reference/analyze.py trl workspace/observed_run` to reconstruct raw correspondence counts. After corpus preparation, set `PC_REPAIR_CORPUS` to the prepared `workspace/data/corpus` and `PC_CASE_PYTHON=/opt/case-venv/bin/python`, then run `python -m pytest -q reference/tests` from this task directory. The latter uses reviewer-only patches and is not an agent-visible success flag.

## Runtime and package checks

The actual `docker/images/tool-rl-vllm024.Dockerfile` was built successfully with Modal's Dockerfile builder. Both final `prepare.sh` scripts completed on that image: all model files passed their revision/SHA-256/size checks, TRL corpus source/test downloads passed their checks, the SQL generator ran, and both frameworks imported. The corpus builder also passed a second, idempotent invocation. A prepared original regression case failed with the expected test counts. These were CPU checks, not a new GPU training campaign or an end-to-end Harbor judge run.

- [Final CPU preparation check](https://modal.com/apps/scale-rsi/improve-vlm-gui-grounding/ap-Suv4HUVqibARLXn4qWMbd3)
- Python 3.12.10, torch 2.11.0, vLLM 0.24.0, transformers 5.10.1, datasets 5.1.0, accelerate 1.15.0, wandb 0.31.0. Legacy repair tests use Python 3.9.25.
- Base-image digest is pinned; the new image's tooling block exactly matches the canonical `trl-grpo-torch26` block. Each asset has a pinned revision, hash, and byte size.
- `python -m autorlbench validate <task>`: zero errors. Canonical issue instruction and verifier files compare byte-for-byte with the benchmark template.
- Source-manifest checks cover every packaged workspace file. Edits to historical artifacts affect labels/paths only; metric values and raw token IDs remain unchanged. No model weights, checkpoints, credentials, or raw W&B binaries are included in the agent workspace.

Repository checks: 64 passed and 3 failed across environment-plan, training-script, and SQL workload tests. All three failures reproduce when the new image is excluded: existing images have an unpinned base, an underscore-containing image tag, and divergent tooling blocks. These pre-existing failures were not changed. The added image passes the corresponding targeted checks.

## Scope

This is an **issue-only** contribution. One paired seed and three optimizer updates per arm verify a production mechanism. They do not establish a learning-performance improvement, calibrated solution threshold, multi-seed reliability, or sealed-test generalization. No judge-model evaluation was run. Full W&B binaries are retained under `reference/raw_wandb`, outside the historical agent workspace. Modal logs are supplementary; the committed traces and metric files contain the evidence needed for diagnosis.
