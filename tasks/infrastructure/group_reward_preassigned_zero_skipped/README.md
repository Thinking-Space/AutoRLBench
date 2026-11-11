# Generator-assigned zero reward lacks group reward metadata

During GRPO training of Qwen3.5-9B on a mixed code-repair corpus, the group filter discards a valid behavioral zero because it lacks binary reward metadata and rejects the entire group.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Code RL |
| **Tags** | `reward-metadata` `premature-zero` `group-filter` |
| **Framework** | miles (GRPO), Megatron training, SGLang rollout |
| **Run time** | ~31 hours (partial run, stopped at iteration 55) |
| **Resources** | 16 × H100 (80 GB) |

## Difficulty

The generator assigns zero to a trajectory that exhausted its turn budget
without submitting a final answer. That zero is a valid behavioral label, but
the group reward adapter skips every sample whose reward is already non-null.
The adapter therefore never writes the reward-mode metadata that the downstream
binary group filter requires.

A representative eight-sample group had eight valid labels: three submitted
answers and five budget-exhausted zeros. The three submitted answers received
ordinary binary group scores, but the five generator-assigned zeros remained
untyped. The filter rejected the otherwise valid group with
`non_binary_reward_mode`.

The recorded log snapshot contains 881 unique groups dropped with this reason.
The same filter also intentionally drops 605 all-pass and 42 all-fail groups;
those are expected dynamic-sampling outcomes and are not evidence for this
defect. The defect is the missing metadata on valid behavioral zeros, not a
lack of model success or an environment failure.

An offline interface regression reproduced the same path from an eight-sample
group. The repaired adapter preserved the real zero, attached a binary reward
record, and retained the group. The source repair passed 31 tests without an
API or GPU request.

Issue-only: the repair was validated offline, but no same-seed, same-budget
post-fix training run is available, so a rerun threshold would be confounded.

## Reference solution

- **issue**: identify that `mark_generator_zero` writes `reward=0.0` before
  group normalization, while `normalize_group` skips preassigned rewards and
  leaves no binary reward record for `accept_valid_group`; reject “all five
  zero samples mean the group is invalid” and “the filter should accept
  arbitrary preassigned rewards”.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  a verdict score of 1.0 and root-cause and evidence scores above 0.8.
