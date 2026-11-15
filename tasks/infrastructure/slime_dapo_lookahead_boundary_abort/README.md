# DAPO lookahead aborts a complete batch on a surplus cleanup failure

Training Qwen3-VL-4B-Thinking with GRPO on CUA-Gym combined 768 tasks loses a complete DAPO update when cleanup of a lookahead candidate outside the consumed prefix raises a shared boundary error.

| | |
| --- | --- |
| **Author** | Zhizhou Wang |
| **Affiliation** | Southwest Jiaotong University |
| **Type** | Infrastructure |
| **Domain** | Tool-use RL · GUI RL |
| **Tags** | `async-dapo` `lookahead-boundary` `cleanup-lifecycle` `batch-abort` |
| **Framework** | slime v0.3.0 · GRPO/DAPO |
| **Run time** | ~2 h 56 min |
| **Resources** | 2 nodes × 8 NVIDIA H100 (80 GB) |

## Difficulty

The final rollout reports 32 accepted groups, yet the optimizer step is lost during the cleanup barrier. The diagnosis requires joining the sample 2889 → group 361 event in the trainer excerpt to the audit rows for rollout 5, whose recorded group indices stop at 358 and whose consumed prefix is only 39 of 56 submitted groups. The relevant behavior spans an unstructured error recorder, two scheduler rethrows at different lifecycle points, and a fenced cleanup exception that is converted into a shared boundary failure. Long cleanup time and high lookahead concurrency are real amplifiers; they do not explain why a surplus failure invalidates an already full batch.

This is issue-only: the historical run has no clean, gradeable control arm for a repair threshold.

## Reference solution

- **issue**: A passing diagnosis must call the run unhealthy and identify all three essential causes: boundary failures are stored without candidate/prefix ownership; the scheduler checks that unowned shared list before prefix consumption and again after the target batch is selected, with the post-selection check discarding the full batch; and an unacknowledged fenced cleanup times out, propagates through `close_session()` and `generate_with_gui.py`, and becomes the shared boundary error. It must explain that the timeout path does not reach `return True`, the `prefix_acceptance_rate` versus `submitted_utilization` denominators, the 53-versus-56 examined/submitted counts, and derive group 361 as outside the 306–358 audit coverage. Cleanup latency/concurrency, reward health, and timeout size are symptoms or amplifiers.

## Verification

- **issue**: The official judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict 1.0 and root-cause and evidence scores above 0.8. Final revised calibration with `deepseek-v4.1-flash-mb` at high reasoning produced passed **0/2**, weighted scores **0.294737 / 0.465789** (root/evidence pairs **0.0/0.473684** and **0.25/0.578947**). Both isolated runs completed without timeout or API failure and reported `input_unchanged=true`.
