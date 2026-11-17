# Response position ids restart at zero: the actor scores rollouts in a view vLLM never sampled

Training Qwen2.5-0.5B-Instruct with GRPO on GSM8K, the agent loop numbers response tokens from 0 instead of continuing after the prompt. vLLM samples with correct positions, but the actor and the reference score every rollout with the response laid over the prompt in RoPE position space. Training reward still climbs and nothing errors; held-out accuracy ends at 0.4776 against 0.5580.

| | |
| --- | --- |
| **Author** | Shulin Tian |
| **Affiliation** | NTU |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | `position-ids` `rope` `train-inference-mismatch` |
| **Framework** | verl 0.8.0 (GRPO), FSDP + vLLM colocated, `run_grpo_gsm8k.sh` |
| **Run time** | ~34 minutes |
| **Resources** | 1 × A100 (40 GB) |

## Difficulty

`verl/experimental/agent_loop/agent_loop.py::AgentLoopWorker._compute_position_ids`:

```python
            prompt_width = self.rollout_config.prompt_length
            if attention_mask.shape[-1] > prompt_width:
                return torch.cat(
                    [
                        compute_position_id_with_mask(attention_mask[..., :prompt_width]),
                        compute_position_id_with_mask(attention_mask[..., prompt_width:]),
                    ],
                    dim=-1,
                )
```

The comment says the left-padded prompt window and the right-padded response window must be
numbered separately. vLLM never reads these tensors (it derives positions from the token sequence),
so rollouts, rewards and validation outputs come from the real policy and re-score cleanly. Only the
actor (old log-probs and update) and the reference read the wrong positions. The importance ratio
and the KL are computed in that same view, so `ppo_kl`, clip fraction and loss look self-consistent.

What shows is indirect: the actor moves far in its own view while held-out accuracy barely moves.

| | defective, seed 42 | defective, seed 43 | control, seed 42 |
| --- | ---: | ---: | ---: |
| held-out step 0 → 60 (re-scored) | 0.4632 → 0.4776 | 0.4511 → 0.4670 | 0.4503 → 0.5580 |
| best held-out | 0.4875 | 0.4890 | 0.5580 |
| `actor/entropy` step 1 → 60 | 0.82 → 0.34 | 0.82 → 0.35 | 0.48 → 0.08 |
| `actor/kl_loss` step 60 | 0.44 | 0.83 | 0.07 |
| `actor/grad_norm` step 60 | 5.1 | 8.6 | 0.9 |
| training reward step 1 → 60 | 0.44 → 0.63 | 0.36 → 0.50 | 0.40 → 0.69 |

Wrong diagnoses: "healthy, a 60-step run gains little" (the control gains 10 points), the format
loophole (the prompt asks for `####`, the flexible scorer takes the last number and the policy drifts
to `\boxed{}`; the control has it too), sampled-versus-greedy sharpening, overfitting, KL too weak,
the 500-character data filter.

## Reference solution

- **issue**: the separate numbering of the response window → actor and reference score rollouts
  under positions that differ from generation → the gradient optimizes a different conditional than
  the one vLLM samples → large drift in the actor's view, little held-out gain. "Train/inference
  mismatch" without the position ids earns partial credit.
- **solution**: number the concatenated window once with `compute_position_id_with_mask`
  (`fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **solution**: the harness reruns `run_grpo_gsm8k.sh` (60 steps, 3 h cap) and the grader re-scores
  `experiment_logs/val_generations/60.jsonl` itself (last number = gold answer, at least 1,300 rows);
  pass at ≥ 0.51. The runner is byte-compared (the fix is in framework code). Defective step 60:
  0.4776 / 0.4670 (best 0.4875 / 0.4890, seeds 42 / 43); control 0.5580 / 0.5406; margins 0.021 to
  the best defective checkpoint and 0.031 to the lower control. Two defective and two control runs:
  margins, not a variance estimate. Grader replay on the recorded evidence: both defective runs
  exit 1, both controls exit 0.

Not verified: a Docker run of the task (the runs used rootless Podman with the host NVIDIA driver on
the same image digest).
