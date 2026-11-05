# TRL GSM8K GRPO with mismatched EOS configuration

Training a GSM8K SFT checkpoint with GRPO inherits mismatched EOS settings:
rollouts continue past completed answers and held-out performance regresses
despite apparently encouraging training reward.

| | |
| --- | --- |
| **Author** | PostTrainBench contributor |
| **Domain** | configuration |
| **Tags** | `trl` `grpo` `gsm8k` `eos` `historical-snapshot` `issue-only` |
| **Model** | Qwen3-1.7B-Base + GSM8K SFT |
| **Dataset** | openai/gsm8k |
| **Framework** | trl, `scripts/run_training.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 1 GPU (issue-only, offline inspection; the count matches the recorded recipe) |
| **Evaluations** | issue |
| **Recorded runs** | A single historical training run (frozen evidence) |

## Difficulty

The generation configuration does not recognize the assistant end-of-turn token
used by the chat template. `scripts/train_grpo.py::main` inherits that setting,
and `correctness` scores the last extracted answer from the resulting completion.
Retained raw rollout records are the strongest direct evidence that generation
continues beyond a completed turn. Training reward alone conceals the problem.

The run exhibits held-out regression rather than demonstrating an absence of
learning. The oracle distinguishes this observed outcome from proof that EOS is
the sole cause. Original metric values remain in frozen machine-readable evidence;
this contribution deliberately does not advertise an accuracy target.
A repaired-training control run with the corrected EOS list (`[151643, 151645]`)
confirms the fix materially improves held-out accuracy relative to the defective
configuration, so the repair is validated by measurement, not mechanism alone.

## Reference solution

- **issue**: identify the EOS/template mismatch and explain its effect on rollout
  termination, answer extraction and training signals using the frozen evidence.
  Distinguish it from unsupported tokenizer or entropy-collapse explanations.

## Verification

- **issue**: the official LLM judge compares `agent_answer.md` with the hidden
  oracle on verdict, root cause and evidence using its unchanged scoring rubric.
- The `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` image is used only to read files and run the standard-library judge;
  it is not asserted to reproduce the historical TRL stack. Historical software
  versions and provenance are in `environment/source_manifest.json`.
- The repaired-training control referenced by the oracle was measured outside
  this packaging.
