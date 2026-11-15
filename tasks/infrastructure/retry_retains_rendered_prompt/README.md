# Retry reset retains rendered prompt and blocks recovery

Historical Qwen3.5-9B GRPO training on a private code-repair corpus completes 160 updates before retained rendered prompts make retry and recycle fail; the public rerun uses a small randomly initialized Qwen2, public color-token inputs and CPU Hugging Face inference with a real Miles Ray/Megatron GPU trainer.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Code RL |
| **Tags** | `retry-lifecycle` `rendered-prompt` `abort-loop` |
| **Framework** | Miles 0.2.1.dev1609+g48f39cb4, GRPO, Ray/Megatron; bundled MCore 0.18.0+caeabd765 |
| **Run time** | Historical master API records span approximately 10 hours 13 minutes |
| **Resources** | Historical: 16 × H100 (80 GB); default public smoke: 1 GPU plus CPU inference; optional SGLang: 2 GPUs, unverified |

## Difficulty

`Sample.reset_for_retry()` preserves `prompt`. Historical generation checks for native messages but later stores rendered training text in that field. Both the complete async producer's same-prompt clone and its source-group recycle reuse that reset. Their next generation therefore encounters a string at a messages-only boundary. The shipped Sample, group scheduler, producer thread/event loop, queues, collector, reset helpers and failure guard are real source, not replacements. The affected overlay is the parent of `5ade2a6`; the original generation boundary is quoted with source lines in `evaluations/issue/historical_contract.txt`.

`workspace/logs/historical/manifest.json` describes the anonymous `historical_failure` evidence: `master-0.jsonl.gz` has 76,899 records and `worker-0.jsonl.gz` has 275. Private payloads become redaction markers while record positions remain. `metrics/step/*.json`, `metrics/rollout/*.json` and `metrics/perf/*.json` retain all 160 step, 20 rollout and 20 performance records with numeric values unchanged. These are the complete records returned by the original training-pod log API for those pods. **Full training-pod API logs do not mean all Ray worker files, inference-server files, or external service logs.** The historical run completed rollouts 0–19 and steps 0–159 before the collector reported 128 consecutive aborted groups; the audited 2,829 normalization-error-bearing raw records include duplicate traceback messages, not 2,829 independent failures.

The new smoke is deliberately smaller. `scripts/train_reproduction.py` launches the archived native `train_async.py`, with a 361,600-parameter randomly initialized public Qwen2, four real generated responses per group and a public token-based reward. Miles owns conversion, GRPO normalization/loss, Ray training actors, Megatron pipeline forward/backward and the optimizer. A hook observes the actual optimizer result, gradient norm and parameter change; it does not replace the update. Only after that evidence exists does a controlled dynamic-filter drop enter the original same-prompt retry path. The existing inflight-based guard is 128 at the smoke's configured inflight limit. The default packaged rerun has now completed 1 native GRPO optimizer update before exiting 1 at the 128-consecutive-abort guard. The reference adapter repair completes 3 native updates and passes both the same-prompt retry and the independent source-group recycle probe.

The private agent, task corpus, prompts, reward service, tools and service environment are replaced with public adapters. The filter deliberately requests one retry using the historical allowlisted reason; it does not measure a solve rate. Default `--generation-backend hf-cpu` uses real CPU Hugging Face inference and one GPU for the native Miles Ray/Megatron trainer. Miles's `--debug-train-only` option skips SGLang servers and actor-to-generator weight synchronization; the configured path retains real Ray actors, GRPO loss, Megatron forward/backward and optimizer updates. The CPU generation model stays at initialization and receives no trained actor weights. Optional `--generation-backend sglang` needs two GPUs and remains unverified: its router did not become ready, so that attempt failed before training. Its configured weight-update interval of 100 would also keep generation weights unchanged between actor updates during this short probe. Neither mode recreates the historical model, rewards, training budget or learning curve. Pure CPU native training is unsupported.

Issue-only: there is no calibrated, matched-budget repaired training comparison or solution-score threshold.

## Reference solution

- **issue**: identify the generation input/output representation conflict, the reset's retained `prompt`, and both original clone/recycle callers. Reconstruct native messages from frozen task fields before generation, while retaining aligned rendered training tokens. The historical fix `5ade2a6` repairs generation reconstruction; the shared reset alone is not an unconditional framework bug. Network, GPU and checkpoint guesses do not explain the proven local type check.

## Verification

- **issue**: the existing judge compares `agent_answer.md` to the hidden oracle using verdict / root cause / evidence weights 0.2 / 0.6 / 0.2. Passing requires verdict = 1.0 and root-cause and evidence scores each > 0.8 under the shipped judge. Task structure, syntax and evidence integrity checks pass. In the default one-GPU/HF-CPU rerun, the actual bundled Miles `train_one_step` and Megatron `ChainedOptimizer.step` complete 1 update with a positive gradient norm and measured parameter change, followed by the expected 128-consecutive-abort failure. The reference adapter repair completes 3 native updates, including same-prompt retry and source-group recycle. Complete sanitized prepare/training logs and observation reports are in `workspace/logs/validation/` and `evaluations/issue/reference/validation/`; `environment/verification.json` records the command and scope. The available platform mirror was tested; the public digest was not executed directly through local Docker. Optional SGLang mode remains unverified after its pre-training router timeout.

The historical 128 is the collector's terminal consecutive-abort count, not a synthetic 128-event archive. No generated smoke events are presented as historical records. Runtime commands, compatibility boundaries and source/license details are documented in `workspace/README.md`, `workspace/runtime/source/README.md` and `workspace/THIRD_PARTY_NOTICES.md`.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The default native failure/reference validation uses the resulting source; its runtime source identities are recorded in the observation reports.
