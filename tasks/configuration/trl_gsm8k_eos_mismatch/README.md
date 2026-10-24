# TRL GSM8K GRPO with mismatched EOS configuration

Training a GSM8K SFT checkpoint with GRPO inherits mismatched EOS settings:
rollouts continue past completed answers and held-out performance regresses
despite apparently encouraging training reward.

| | |
| --- | --- |
| **Author** | PostTrainBench contributor |
| **Domain** | configuration |
| **Tags** | `trl` `grpo` `gsm8k` `eos` `historical-snapshot` |
| **Model** | Qwen3-1.7B-Base + GSM8K SFT |
| **Dataset** | openai/gsm8k |
| **Framework** | trl, `scripts/run_training.sh` |
| **Environment** | `verl`, offline inspection and configuration repair; no GPU needed |
| **Evaluations** | issue · solution (configuration-repair variant) |
| **Recorded runs** | A single historical training run; no matched repaired-training control |

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

## Reference solution

- **issue**: identify the EOS/template mismatch and explain its effect on rollout
  termination, answer extraction and training signals using the frozen evidence.
  Distinguish it from unsupported tokenizer or entropy-collapse explanations.
- **solution**: add the assistant end-of-turn token to the EOS list, retaining
  the existing end-of-text token and all other generation configuration fields.

## Verification

- **issue**: the official LLM judge compares `agent_answer.md` with the hidden
  oracle on verdict, root cause and evidence using its unchanged scoring rubric.
- The `verl` image is used only to read files and run the standard-library judge;
  it is not asserted to reproduce the historical TRL stack. Historical software
  versions and provenance are in `environment/source_manifest.json`.
- **solution**: a custom deterministic verifier checks the workspace EOS list,
  unchanged non-EOS fields and synthetic token-stream stopping boundaries.
  This configuration-repair variant does not meet the official repair-and-rerun
  evaluation contract; upstream acceptance of this variant is not established.
- No weights or dataset downloads are needed. There is no repaired-training
  control or calibrated training-success threshold. Passing the solution checks
  establishes a configuration edit, not improved training or model inference.
- Structural validation and local checks are recorded by the submission guide.
  Container warm-cache preparation, external LLM judging and GPU training have
  not been verified as part of this packaging.
