# Flat image-token positions break training-rollout parity

Training Qwen3-VL-8B-Instruct with GRPO on Geometry3K. verl's training path builds its processor with a raw `AutoProcessor.from_pretrained` instead of the `hf_processor` factory, so image tokens get flat positions in training while vLLM assigns the correct multimodal positions — the actor and the rollout engine assign different log-probabilities to the same response tokens, and validation plateaus below the control.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Multimodal Math RL |
| **Tags** | `flat-rope-positions` `train-inference-parity` `processor-init` `silent-failure` |
| **Framework** | verl v0.9.0 (GRPO), FSDP2 + vLLM 0.24 / torch 2.11 / CUDA 13 |
| **Run time** | ~1 hour per run (~39 minutes of training steps alone) |
| **Resources** | 8 GPU |

## Difficulty

The defect is in `verl/workers/config/model.py::HFModelConfig.__post_init__`: the workspace constructs the processor with raw `AutoProcessor.from_pretrained` instead of verl's `hf_processor` factory. The raw processor lacks the dynamically bound `get_rope_index` capability, so the agent-loop fallback assigns flat positions to image sequences, while vLLM independently constructs the correct multimodal positions — the actor and the rollout engine therefore assign different log-probabilities to the same response tokens.

Nothing fails loudly: the run completes all 20 steps with finite gradients, plausible training reward (0.548 vs 0.545 control), and validation improves (57.07% → 59.07% binary). The single strongest piece of evidence: the frozen parity measurement — on six frozen inputs captured before training (64 response tokens per case), the maximum absolute response-token log-probability disagreement between the training and rollout paths is 3.74 nat, against 0.55 nat on the control. Reading traps: `actor/kl_loss` compares against the reference policy, not the rollout engine; `training/rollout_probs_diff_*` measures probability differences, not nats; the validation log's `acc` field carries the `0.9 * correctness + 0.1 * format` reward, not binary accuracy.

| Recorded metric (20 steps, 8 H800) | Defective | Control |
| --- | ---: | ---: |
| `rollout_corr/kl`, nat/token | 0.006591 | 0.000623 |
| Token-probability Pearson correlation | 0.993060 | 0.999253 |
| Frozen max abs log-prob disagreement, nat | 3.740287 | 0.550462 |
| Step-20 binary accuracy (601 held-out) | 59.07% | 61.56% |

## Reference solution

- **issue**: a passing diagnosis names `HFModelConfig.__post_init__` and the raw `AutoProcessor.from_pretrained` construction — the lost `get_rope_index` capability leaving image sequences with flat positions in training while vLLM computes the correct multimodal positions — and connects it to the training–rollout log-probability disagreement. Attributions that must fail: BF16 numerics, GRPO or the reward as the cause, and an absence of learning (validation improves on both arms).
- **solution**: restore the upstream processor initialization — call verl's `hf_processor` factory in `HFModelConfig.__post_init__`, preserving `get_rope_index`. Sufficient because relative to the pinned upstream commit `483b8a0` the workspace replaces only the processor factory; the model, dataset, reward, launch budget, and validation protocol stay unchanged.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden `oracle.md` on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing is a conjunctive gate, not the weighted sum: `verdict_score == 1.0` and `root_cause_score > 0.8` and `evidence_score > 0.8`.
- **solution**: the agent repairs the workspace and submits `report.md` without launching training; the harness reruns the 20-step experiment per `run_spec.json` (4 h timeout, 8 GPUs) and requires finite nonzero gradients, repeated validation, offline W&B evidence, and freshly captured parity for all six frozen cases — every case must have correct positions and maximum absolute response-token log-probability disagreement below **1.0 nat**. Calibration: the recorded control maximum is 0.550462 nat (0.4495 below the limit), the defective maximum 3.740287 nat (2.7403 above); offline replay rejects the recorded defective side and accepts the control. Margins only, no variance — one run per side, and the step-0 validation already differs between the arms, so the final accuracy gap is not entirely attributable to the repair. The `verl-qwen3vl-cu130` image is not yet published; a fresh GPU experiment in the declared image is pending.
