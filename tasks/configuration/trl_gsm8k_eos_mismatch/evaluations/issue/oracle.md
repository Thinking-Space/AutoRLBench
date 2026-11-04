# Canonical issue oracle

Reconciled from the historical GRPO source, original checkpoint metadata,
training log, retained rollout records and before/after diagnostic evaluations.
The defective snapshot has no in-package control; a separately measured
repaired-training control with the corrected EOS list is summarized below.

Expected verdict: **unhealthy**.

## Root cause

- `model/generation_config.json::eos_token_id`,
  `scripts/train_grpo.py::main` and `templates/qwen3.jinja`
  - kind: `hparam`
  - observed: the saved EOS list recognizes `<|endoftext|>` but not `<|im_end|>`;
    the conversation template closes assistant turns with `<|im_end|>`.
    Training retains this setting rather than supplying an aligned EOS override
    to the generation backend.
  - expected: rollout termination recognizes the actual assistant end-of-turn
    marker. A completed answer should not continue until the completion budget
    merely because the backend does not recognize its turn boundary.

## Observable impact

- `outputs/rollout_samples.jsonl` retains special tokens and shows continued
  generation after the assistant turn-ending marker, together with recorded
  termination status and long/repetitive completions.
- `outputs/training.log` and `outputs/training_metrics.json` show completion
  clipping and groups with uninformative reward variation, while training reward
  alone can appear encouraging.
- `scripts/train_grpo.py::correctness` extracts the last numeric answer from a
  completion. Continued generation can change the scored answer or waste the
  rollout budget. The extra monitoring rewards have no optimization weight.
- `outputs/evaluation_metrics.json` records poorer held-out performance after
  this training run under a common diagnostic evaluation protocol. These are
  actual recorded measurements, not a performance threshold for this tier.

## Trigger conditions

- The SFT conversation uses an assistant end-of-turn token that is absent from
  the inherited generation stopping configuration.
- GRPO samples with that configuration and scores the resulting full completion.
- Training reward is mistaken for evidence that the held-out task improved.

## Documentation reconciliation

- The concrete, demonstrated defect is stopping-token misalignment. The observed
  held-out regression is consistent with a harmful rollout/reward interaction.
  A repaired-training control retraining with the corrected EOS list
  (`[151643, 151645]`) was measured to materially improve held-out accuracy,
  so the fix's effect is validated by measurement; the regression should still
  not be attributed to EOS as the sole cause of every optimization change.
- The before/after held-out measurements use aligned stopping and greedy decoding
  for both checkpoints. They are diagnostic evaluations, not a repaired GRPO run.
- Do not claim that EOS alone restores a particular accuracy or that no learning
  occurred. Decoding settings affect evaluation independently of stopping.
- Do not require a monotonic entropy collapse, uniformly zero gradients, or a
  particular score change: those claims are not established by this snapshot.
- Dataset-cache notices, generic tokenizer warnings and shutdown warnings are not
  sufficient root-cause evidence. A good diagnosis connects effective settings,
  actual token-level behavior and optimization/held-out symptoms.
