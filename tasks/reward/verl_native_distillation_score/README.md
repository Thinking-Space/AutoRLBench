# Math accuracy slips while training perplexity falls

During on-policy distillation (OPD) of Nemotron-Research-Reasoning-Qwen-1.5B on Skywork math, the student's AIME24 score drops from 54 to 36 correct answers out of 240 between steps 30 and 60, while the perplexity logged during training falls.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | distillation, heldout-regression, math |
| **Framework** | VeRL, vLLM 0.24.0 |
| **Run time** | About 110 minutes for steps 31–60, after the initial 30 steps |
| **Resources** | 2 GPUs |

## Difficulty

The student generates answers in vLLM with a quantized KV cache and updates its weights in VeRL. For each next-token prediction, the distillation reward compares the student with a frozen teacher on the union of their top-512 candidates. The run uses 32 Skywork prompts per step with one answer per prompt and an 8,192-token limit.

The separate AIME24 evaluation uses the same 30 problems at every checkpoint, with eight attempts per problem and a 32,768-token limit. On one problem whose answer is 540, the student goes from eight correct attempts at step 30 to four at step 60. All 240 answers at each checkpoint finish naturally. Meanwhile, the metric logged as `training_ppl` falls from 1.98 at step 31 to 1.20 at step 60. This metric measures how predictable the sampled tokens are to vLLM; AIME24 measures whether the completed answer is correct.

## Reference solution

- **issue**: For tokens proposed by the student, the reward reuses the student probabilities from vLLM. For tokens added by the teacher, it takes student probabilities recomputed by the training worker. It joins the two sets of scores, then uses them both to compare against the teacher and to weight the reward. This makes the learning signal depend on which model proposed a candidate. The repair keeps sampling and candidate selection as they are and scores every candidate with the same student forward pass. Both runs score 54/240 at step 30; the repaired run reaches 63/240 at step 60, 27 correct answers ahead of the faulty run.

## Verification

- **issue**: The judge grades `agent_answer.md` against the hidden oracle. The workspace includes logs through step 60 and all 1,440 evaluation answers from six checkpoints. The bundled launcher runs the first 24 steps; the later steps are documented by the recorded run.

The task is Issue-only. The step-60 comparison comes from one training run per arm.
