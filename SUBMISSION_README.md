# Seven-case GRPO contribution

This tree is based on upstream259ebf12c2f2b78c60f7500e124115c0aaeae4a5. Seven candidate cases share ONE repaired task2 run sbr1bh7d. Eight original training runs completed150 steps and passed original checkpoint validation. No additional training or PR publication was performed during packaging.

Completed source/data work: a digest-pinned x86_64 vLLM0.11/PyTorch2.8 image definition;217 recorded package pins and SHA-pinned FlashAttention wheel; real public model/data preparation with exact derived parquet hashes; isolated foreground rerun launcher; protected grading contract, independent answer rescoring and checkpoint archive checks; applicable reference patches; seven official per-task validators; native Hydra and verifier replay tests.

Container build/warm-cache and GPU reruns were explicitly deferred by the user because no Docker/Podman GPU host is available. They are NOT claimed passed. Local two-Ray-node CPU isolation does not prove GPU partitioning or physical-node equivalence. The historical two-node deployment and portable one-host/two-logical-node deployment must be distinguished. Live LLM diagnosis judging is also untested.

## Inspect and validate source
Run `python3 submission_validation/verify.py` from this root. It does not start training, use a GPU or download assets. See submission_validation/summary.json for receipts and the known upstream test discrepancy.

## On a container host, after review
1. Build the new image: `python3 -m autorlbench build-images verl-v061-grpo` uses the new Dockerfile and docker/ context.
2. Run `bash docker/warm-cache.sh <task-id>` to validate container preparation. The pinned model is about4GB; raw train data about299MB. These binaries are deliberately not in Git.
3. Use `python3 -m autorlbench harbor-shell --help` or the documented native evaluator to prepare a staged workspace. Apply the hidden reference patch only to a reviewer-controlled verification copy, add a report, and run a clean harness solution evaluation with150 steps.
4. For exact physical topology, provide two shared-path Ray GPU workers with4 GPUs each through RAY_ADDRESS. Otherwise the foreground runner creates two logical workers inside one8-GPU container. The latter needs a fresh acceptance run.
5. Record the built image digest, warm-cache receipts and GPU results before claiming end-to-end reproduction. Existing shared reference training evidence remains valid and is not multiplied into seven runs.

## Interpretation
Task1 measures reward recovery despite no AIME regression. Task3's mask-only variants are deviations from an explicitly declared standard-GRPO comparison contract, not universally invalid objectives. Related variants are not automatically seven independently accepted tasks. The task2 reference supplies baseline metrics for all cases; task4-specific repaired sync telemetry is absent.

## Files for an eventual PR
Add the seven task directories; docker/images/verl-v061-grpo.Dockerfile; docker/verl-v061-runtime; the docker/.dockerignore allowlist addition; submission validation tests/docs and shared baseline reviewer evidence as agreed with maintainers. No credentials, models, parquet data or checkpoint binaries are included.

## Maintainer pre-review requests
These recorded runs used SwanLab, console/file logs and JSONL metrics, not offline W&B. No run-*.wandb binary is supplied. Please confirm whether the supplied stdout/stderr, per-step metrics, rollout samples and SwanLab evidence are acceptable in place of the CONTRIBUTING.md W&B artifact requirement. This exception is not claimed approved.

Please also review shared-reference eligibility, the four related task3 objectives against the explicitly declared standard-GRPO target, and whether the cases should be grouped. Container/GPU verification is pending; this contribution is a draft for pre-review.
