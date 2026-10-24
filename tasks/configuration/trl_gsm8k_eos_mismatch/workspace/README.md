# Recorded GRPO experiment

This is an offline historical diagnosis snapshot of a GSM8K GRPO run initialized
from an SFT checkpoint. The reported problem is degraded held-out performance
despite apparently useful training reward. Inspect source and effective settings
together with the recorded outputs; do not infer model health from training reward alone.

- `scripts/train_grpo.py`: recorded training/reward logic with portable path defaults.
- `scripts/run_training.sh`: normalized in-container launch entry point.
- `model/`: original model, generation and tokenizer metadata; no weights.
- `templates/`: template used for training prompts.
- `outputs/training.log`: recorded run log, with machine identifiers sanitized.
- `outputs/training_metrics.json`: complete recorded trainer log history.
- `outputs/rollout_samples.jsonl`: unchanged early, middle and late rollout records.
- `outputs/evaluation_metrics.json`: before/after diagnostic held-out evaluations
  using the same aligned stopping and greedy decoding protocol.

The launch script documents the historical training recipe, not a supported rerun
in the diagnosis image. Model weights, full tokenizer vocabulary and the training
dataset are deliberately not staged: this tier requires neither inference nor
training. Do not launch training. The issue tier treats this snapshot as read-only;
the configuration-repair solution tier permits only the edit described in its
instruction. Frozen outputs remain read-only evidence in either tier.
