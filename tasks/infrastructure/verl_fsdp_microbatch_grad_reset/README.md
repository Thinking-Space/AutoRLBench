# Gradient reset inside the micro-batch loop keeps 1/8 of every mini-batch

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, verl's FSDP engine clears gradients before every micro-batch's backward, so only the last of 8 micro-batches reaches the optimizer. AdamW still takes full-size steps, so the policy drifts like a healthy run, but along a direction estimated from 4 prompt groups. Training reward rises; held-out accuracy ends below its starting value (0.4511 vs 0.4610) against 0.5580 for the same recipe.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | NTU |
| **Type** | Infrastructure |
| **Domain** | Math |
| **Tags** | `gradient-accumulation` `fsdp` `noisy-updates` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~34 minutes |
| **Resources** | 1 × A100 (40 GB) |

## Difficulty

`verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.forward_backward_batch`:

```python
                if not forward_only:
                    # Release the gradient buffers left from the previous micro-batch before this backward so that
                    # peak memory stays at one micro-batch of gradients plus the sharded parameters.
                    self.module.zero_grad(set_to_none=True)
```

The recipe accumulates over micro-batches (`ppo_mini_batch_size=32` prompts × `rollout.n=8` = 256
samples, `ppo_micro_batch_size_per_gpu=32`), so 7/8 of each batch's rollouts never reach the
optimizer. Nothing errors, rewards re-score cleanly, and the drift metrics look like a healthy run
because AdamW normalizes the step size. The only direct trace is a gradient norm about 3× smaller
than usual, which has no reference inside the workspace.

| | defective, seed 42 (shipped) | control, seed 42 |
| --- | ---: | ---: |
| held-out step 0 → 20 → 60 (re-scored) | 0.4610 → 0.4723 → 0.4511 | 0.4503 → 0.4973 → 0.5580 |
| `actor/entropy` step 1 → 60 | 0.49 → 0.10 | 0.48 → 0.08 |
| `actor/kl_loss` step 60 | 0.083 | 0.070 |
| `actor/grad_norm` range | 0.12-0.36 | 0.58-0.94 |
| training reward step 1 → 60 | 0.39 → 0.60 | 0.40 → 0.69 |

Wrong diagnoses: "healthy, a 60-step run gains little", the format loophole (the prompt asks for
`####`, the flexible scorer takes the last number and the policy drifts to `\boxed{}`; the control
has it too), over-optimization or entropy collapse, the 500-character data filter.

The defect's outcome is seed-sensitive: a second defective run (seed 43) ends at 0.5057, against
0.5406 for the seed-43 control. That gap is too small for a fixed threshold, so this task ships the
issue level only.

## Reference solution

- **issue**: the reset inside the micro-batch loop → only the last micro-batch's gradient survives →
  each update is estimated from 32 of 256 samples → full-size AdamW steps in a noisy direction →
  held-out accuracy does not improve. "Noisy updates" or "small gradient norm" without the reset
  earns partial credit.
- Fix for reference: remove the reset (`evaluations/issue/reference/fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- Recorded comparison: `evaluations/issue/reference/RESULTS.md` (control runs are those of
  `tasks/infrastructure/verl_fsdp_bf16_master_weights`).

Not verified: a Docker run of the task (the runs used rootless Podman with the host NVIDIA driver on
the same image digest).
