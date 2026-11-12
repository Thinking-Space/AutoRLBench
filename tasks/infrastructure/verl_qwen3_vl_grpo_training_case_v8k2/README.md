# Fused sequence-parallel GRPO trains on misaligned next-token labels

Training Qwen3-VL-8B-Instruct with GRPO on ScienceQA image multiple choice (2560 train / 1024 validation, seed 42). In the fused FSDP/Ulysses path the model adapters roll their own labels instead of consuming the engine's globally shifted, SP-sliced labels, so wrong-token positions enter the response loss mask and receive nonzero advantages and gradients — training stays silent while validation drifts down.

| | |
| --- | --- |
| **Author** | Yuan Gao |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Multimodal RL |
| **Tags** | `label-misalignment` `sequence-parallel` `fused-ppo-loss` `silent-failure` |
| **Framework** | verl @ `4b9c14f2` (GRPO), FSDP4 FULL_SHARD DP2×SP2, fused/remove-padding, torch 2.8 / vLLM 0.11 / cu129 |
| **Run time** | ~6.6 hours |
| **Resources** | 4 GPU |

## Difficulty

The defect is in the fused FSDP/Ulysses path. The engine prepares globally rolled and SP-sliced next-token labels in `output_args["input_ids_rmpad_rolled"]` (`framework/verl/workers/engine/fsdp/transformer_impl.py`), but the fused model adapters — `framework/verl/models/transformers/qwen3_vl.py::forward_with_torch_backend` and `FusedLinearForPPO` in `framework/verl/utils/experimental/torch_functional.py` — generate their own rolled labels from their input/labels rather than consuming the prepared tensor. In the measured multimodal execution the effective label extent can exceed the local hidden-state extent, and the fused loop consumes its first `T` labels.

Nothing fails loudly: both formal runs complete 40 updates with exit 0 and finite gradients, training reward stays plausible (first-20 0.951, last-20 0.918), and the validation decline is mild enough to pass the frozen ≤ −5 pp quality-drop criterion (last-5 change −2.58 pp) — quality trends alone do not flag it. The strongest evidence is the paired instrumentation: response-label mismatches 16,311,325, nonzero-advantage mismatches 2,896,350, and wrong-position nonzero gradients 381,610 on the defective arm (capped repeated observations, not unique token counts), against 0 / 0 / 0 on the repaired arm. A correct explanation relates next-token labels, sequence partition, response loss positions and the PPO gradient.

| Recorded metric (40 updates, 4 GPU, one run per arm) | Defective | Repaired |
| --- | ---: | ---: |
| Response-label mismatches | 16,311,325 | 0 |
| Nonzero-advantage mismatches | 2,896,350 | 0 |
| Wrong-position nonzero gradients | 381,610 | 0 |
| Validation step 0 → 40 (`val-core/scienceqa_image_mc/acc/mean@1`) | 0.9385 → 0.9004 | 0.9463 → 0.9727 |
| Last-5 validation change | −2.58 pp | +2.77 pp |

## Reference solution

- **issue**: a passing diagnosis names the fused FSDP/Ulysses label path — the adapters' locally rolled labels versus the engine-prepared `input_ids_rmpad_rolled` (globally shifted, SP-sliced) — and connects the wrong-label positions to the response loss mask, nonzero advantages, and PPO gradients. Attributions that must fail: reward extraction, data leakage, NaNs or optimizer crashes, and the unverified reduction to "exactly one wrong token per rank".

Issue-only: no solution tier is declared — the paired evidence is single-seed on a public substitute for the source issue's internal dataset, and the reference patch stays hidden material. That patch (from upstream verl PR #6268, against base commit `4b9c14f2` recorded from issue #6068) passes the engine-provided `shift_labels` into the five model-adapter files with precedence over the local roll; with it, the paired mismatch counts become 0 / 0 / 0 and the paired validation-change improvement is +5.35 pp.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.

The task runs on the packaged `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` image (ghcr.io/yhyayi/autorlbench, digest-pinned). The original recording ran on the author workstation's cu129 conda env (full freeze preserved in `environment/recorded-runtime-lock.txt`); the cu128 image matches that stack version-for-version on every live compiled layer (vLLM 0.11.0, torch 2.8.0, triton 3.4.0, xformers 0.0.32.post1, transformers 4.57.6, nccl 2.27.3), with `prepare.sh` shadowing numpy 1.26.4 + scipy 1.17.1 (verl pins numpy<2.0.0). **Re-record pending**: the paired forty-step evidence below was captured on the cu129 workstation stack, and the paired evidence and the frozen figures cited above (mismatch counts, validation-change deltas, the −5 pp threshold) still need a full re-record on this image before they can be treated as its ground truth. The defective arm has been smoke-verified on this image (2026-11-11, 4×H100 Cybertron devspace): a 2-update run completed both GRPO cycles — vLLM rollout, regex reward, FSDP/Ulysses `update_actor` with weight sync back to vLLM — and saved the `global_step_2` checkpoint on all four ranks at ~3.5 min/step; the step-0 validation metric (0.9394) matched the recorded value (0.9385). Note the run needs ≥ 400 GiB host RAM (param+optimizer offload peaks above 204 GiB) and the platform reclaimed the verification devspace during wrap-up, so a clean exit-0 was not observed.
