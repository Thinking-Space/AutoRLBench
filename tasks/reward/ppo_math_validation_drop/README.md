# Unexpectedly low math validation score in PPO

At step 500 of PPO training on DAPO-Math-17k, DeepSeek-R1-Distill-Qwen-1.5B received credit for only 924 of 3,200 held-out math answers, and several answers that matched the key earned zero.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Reward |
| **Domain** | Math RL |
| **Tags** | `validation` `ppo` |
| **Framework** | VeRL V1 synchronous PPO |
| **Run time** | 4 h 6 min for the recorded continuation |
| **Resources** | 2 GPUs |

## Difficulty

The run resumed at step 475. At the next scheduled evaluation, the model generated 32 answers for each of 100 problems from AIME 2024, AIME 2025, and AMC 2023. A zero counted as an incorrect answer in the accuracy metric. In one AMC response, the model ended with `\boxed{5}`, the answer key was 5, and the recorded score was zero.

The optimizer continued updating on either side of the evaluation. The clue was in the validation log: a single grading worker reported 1,300 errors. Regrading the same 3,200 answers against the same keys raised accuracy from 28.88% to 49.66% (1,589 correct). The workspace includes the original logs and one complete answer per problem.

## Reference solution

- **issue**: The validation worker uses MathVerify to grade answers in a pool of child processes. After one child exited, the worker kept sending answers to the unusable pool. Each failed grading request returned the same zero used for an incorrect answer. DAPO training rewards followed a separate path that checked boxed answers. The log establishes what happened after the child exited; its initial exit cause is unknown.

## Verification

In the shared image, all 3,200 answers received the same labels as in the independent regrade. A controlled replay isolated the cause: a healthy pool marked 50 of 100 archived answers correct. After one grading child was stopped, the same pool marked all 100 wrong. A fresh pool restored every original judgment. The official harness also completed two PPO updates from the public base weights. The step-500 evidence comes from the archived continuation.

- **issue**: The judge scores `agent_answer.md` against the hidden diagnosis. Verdict, root cause, and evidence carry weights of 0.2, 0.6, and 0.2. The passing score is 0.8. This issue evaluates whether the solver can explain the scoring failure; the archive provides no post-repair training accuracy target.
