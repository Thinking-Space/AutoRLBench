# Granite GSM8K GRPO run

Granite-3.3-2B-Instruct is trained with GRPO on the full GSM8K training set. Every gradient-accumulation micro-step reuses the same 4 of the 64 scored rollouts, so the run still improves but ends at 79.8% held-out accuracy instead of 82.8%.

| | |
| --- | --- |
| **Author** | Haoyang Su |
| **Affiliation** | Fudan University; Shanghai Innovation Institute |
| **Type** | Algorithm |
| **Domain** | Math RL |
| **Tags** | `gradient-accumulation` `rollout-buffer` `effective-batch-size` `silent-failure` |
| **Framework** | TRL (GRPO), commit `d2edcfebe52b1d0f4c0e15dc6a27277105b92b1f` with the upstream Granite logit-scaling fix |
| **Run time** | ~41 minutes |
| **Resources** | 1 × H100 (80 GB) |

## Difficulty

`GRPOTrainer._prepare_inputs` generates and scores 64 rollouts per optimizer step, shuffles them and splits them into 16 micro-batches of 4, one per accumulation micro-step. The buffer is indexed with `self.state.global_step` instead of the micro-step counter `self._step`. `global_step` does not change inside an optimizer step, so the same 4 rollouts are accumulated 16 times and the remaining 60 never reach the loss. Regeneration still keys on `self._step`, so a fresh batch is generated and scored every step. The defect is staged as this single line in an otherwise unmodified TRL tree.

Nothing looks broken. Rewards, advantages and every logged rollout statistic are computed over all 64 rollouts, the trained completions keep their own advantages, and the run improves on both training reward and held-out accuracy. The gap only shows against what the same recipe should reach:

| Held-out accuracy (1319 questions, greedy) | Step 0 | Step 20 | Step 40 |
| --- | ---: | ---: | ---: |
| Defective | 58.7% | 78.4% | 79.8% |
| Control (index repaired) | 58.7% | 79.0% | 82.8% |

Training reward over the first and last five steps is 0.631 → 0.859 for the defective run and 0.713 → 0.894 for the control. Both runs share the step-0 model, data order, seed and first rollouts; the only source difference is the index expression. A diagnosis that reads the improving curve as healthy, or blames the learning rate, KL coefficient or GRPO sample efficiency, misses the cause.

## Reference solution

- **issue**: name the buffered micro-batch index in `_prepare_inputs` (optimizer step used instead of the micro-step counter, so one 4-rollout micro-batch is reused for all 16 accumulation steps) and connect it to the format-only held-out gain, the non-zero sign-flipping loss and the erratic gradient norms. A healthy verdict, hyperparameter explanations, reward or data issues and advantage misalignment are wrong. The repair is `self._step % self.args.steps_per_generation`.

Issue-only: the held-out gap is 3.0 points from one run per side, too thin for a calibrated repair threshold.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. In three fresh-context trials on the packaged workspace (Claude Opus, Sonnet and Haiku), one identified this root cause; the other two judged the run healthy.
