# Quantized GRPO scores depend on batch layout

DeepSeek-R1-Distill-Qwen-1.5B is trained with GRPO on DAPO-Math-17k, where changing left padding alters the actor's log-probabilities and policy gradients for the exact same response.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `attention-mask` `layout-sensitivity` `policy-gradient` |
| **Framework** | PyTorch 2.11.0, Transformers 5.5.3, vLLM 0.24.0 (GRPO) |
| **Run time** | ~15 minutes (shared-image reproduction) |
| **Resources** | 1 GPU |

## Difficulty

Although the actor supplies an attention validity mask and aligned position IDs, its custom attention implementation is registered only in the attention-function registry and is missing from Transformers' separate mask-builder registry (`ALL_MASK_ATTENTION_FUNCTIONS`). Because mask preprocessing does not recognize the custom attention name, it returns `None`. Qwen2 passes `None` into the custom attention wrapper, which forwards it directly to SDPA, allowing real-token queries to attend to preceding padding keys even though the position IDs keep real-token positions aligned. Training still completes rollouts and performs finite optimizer updates without raising an error.

To isolate this defect, the shared-image reproduction runs one optimizer step on 16 prompts with 8 responses each (a 128-response batch), seed 42, and a 16,384-token output limit. Differing only in mask registration (aside from placement and resource time limits), both the defective arm and the mask-restored control generated the exact same 128 response token sequences, scored 54/128 correct before the update, and completed a full optimizer step with saved model, optimizer, random state, and data position.

A mask-only control isolates the forward pass further by re-scoring a single 5,789-token response across two left-padding layouts with the quantizer temporarily disabled and real-token positions held fixed:

| Arm | Same-token logprob MAE | Gradient norm |
| --- | ---: | ---: |
| Defective | 0.28738564 | 0.12818459 |
| Mask registration restored | 0.00721349 | 0.07949457 |

In the fixed-width forward pass, the defective arm passes no mask to attention or SDPA, whereas the restored arm supplies a boolean mask excluding all 974 padding keys. Visible evidence in `workspace/evidence/shared_image/` includes the defective metrics, scalar logs and three original responses. A public observation script also inspects the same dispatch behavior on a small Qwen2 model without training.

The repository also retains a historical 60-step run (step-60 padding-effect MAE of 42.54288; final evaluation 232/960) and a separate discrepancy between native rollouts and unpadded Hugging Face prefills, which reflects a different path mismatch rather than the within-actor layout defect. Because the historical mask-repair evaluation scored 225/960, this task is issue-only and sets no solution accuracy threshold.

## Reference solution

- **issue**: Identify the missing registration in `ALL_MASK_ATTENTION_FUNCTIONS`, trace the resulting `None` mask through the custom attention wrapper to SDPA, explain why aligned position IDs alone leave padding keys visible, and connect this scoring error to the policy-gradient computation. A well-supported diagnosis of the independent native-versus-HF prefill discrepancy can earn additional credit, while remaining distinct from the within-actor left-padding bug.

## Verification

- **issue**: An LLM judge compares `agent_answer.md` against the hidden oracle on verdict, root cause, and evidence (reporting weights 0.2 / 0.6 / 0.2). Passing requires a verdict score of 1.0 and both root-cause and evidence scores strictly above 0.8. Grading evaluates only the diagnosis and does not rerun training. The official CLI validator passes, all four evaluation scaffold files match the current template byte for byte, and warm-cache preparation ran completely in the declared fixed image—rebuilding 17,259 training questions and 30 evaluation questions with hashes matching the recorded reproduction inputs, alongside full checkpoints from both shared-image runs.
