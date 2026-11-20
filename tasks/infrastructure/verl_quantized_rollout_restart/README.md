# Math rollouts exhaust the output budget

During VeRL online policy distillation of Nemotron-Research-Reasoning-Qwen-1.5B on Skywork math, the rollout worker reports a successful weight refresh, then prints a blank answer. Every response in the three recorded batches hits the 16,384-token limit.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math RL |
| **Tags** | rollout-collapse, quantized-inference, long-generation |
| **Framework** | VeRL, vLLM 0.24.0 |
| **Run time** | About 31 minutes for the faulty task run, including checkpoint preparation |
| **Resources** | 2 GPUs |

## Difficulty

The first logged example asks for the repeating part of the decimal expansion of 1/49. The response is a series of blank lines. Each step asks 64 Skywork questions and generates one answer per question. The failure is visible in the first batch, before any policy update, and the original run keeps the learning rate at zero.

The workspace includes the original launch and logs from three runs: one that fails at startup, one where most answers end normally, and one that starts successfully but produces blank samples. The key is to compare what the trainer and rollout worker do between startup and the first answer.

On the shared image, all 192 responses across three batches hit the token limit. Preserving the quantization scales across sleep and wake reduced that count to 8 out of 192 with the same model and prompts. The rerun uses the archived trainer and reconstructs the packed weights byte for byte from the original calibration inputs. (The 8/192 control count is from the author's records; the shipped evidence carries the defective rerun and the historical runs, not the scale-preserving control run.)

## Reference solution

- **issue**: The faulty run loses its frozen quantization scales when the rollout worker releases its weight storage. The next refresh repacks the weights with invalid scales. The earlier run keeps the weight storage in place, so the scales survive.

## Verification

- **issue**: The harness grades `agent_answer.md` against the hidden oracle. The task focuses on diagnosing the generation failure from the recorded runs. The training script reproduces the original three steps on the declared shared image.
