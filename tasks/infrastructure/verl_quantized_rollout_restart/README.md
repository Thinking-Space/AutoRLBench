# Math rollouts exhaust the output budget

During VeRL online policy distillation of Nemotron-Research-Reasoning-Qwen-1.5B on Skywork math, the rollout worker reports a successful weight refresh, then prints a blank answer. Every response in the three recorded batches hits the 16,384-token limit.

| | |
| --- | --- |
| **Author** | Yong Yan |
| **Affiliation** | Tsinghua University |
| **Type** | Infrastructure |
| **Domain** | Math |
| **Tags** | rollout-collapse, quantized-inference, long-generation |
| **Framework** | VeRL, vLLM 0.24.0 |
| **Run time** | About 31 minutes for the faulty task run, including checkpoint preparation |
| **Resources** | 2 GPUs |

## Difficulty

The trainer and rollout worker share the same GPUs. The rollout worker releases memory between batches, then wakes up and reloads weights from the actor before generating again. In the first recorded batch, the model is asked for the repeating part of the decimal expansion of 1/49. The log shows a successful weight refresh followed by a series of blank lines.

The workspace includes the original launch and logs from a startup failure, a run with normal answers, and the run that produces blanks. Comparing those runs helps narrow the search to what happened between loading the model and generating the first answer.

## Reference solution

- **issue** The worker's level-2 sleep discards four frozen NVFP4 calibration tensors along with its weights. The next refresh reloads and repacks weights using those calibration values, so a successful refresh can still leave the sampler with invalid weights.
- **solution** Register the four calibration tensors as buffers before the first level-2 sleep so that vLLM saves and restores them. The worker can still release its weight storage and refresh from the live actor when it wakes. See `evaluations/solution/oracle/fix.diff`.

## Verification

- **issue** The judge checks the diagnosis in `agent_answer.md` against the reference and the recorded runs.
- **solution** The harness runs three batches of 64 prompts on two GPUs. To pass, at least 48 responses per batch must stay below 16,384 tokens. The verifier reads the native VeRL metrics, checks that level-2 sleep released memory, and requires a complete step-3 checkpoint. The learning rate stays at zero, which isolates generation after wake from learning.

| Configuration | Responses at the limit, step 1 | Step 2 | Step 3 | Verifier |
| --- | ---: | ---: | ---: | --- |
| Original | 64/64 | 64/64 | 64/64 | Fail |
| Preserve the calibration tensors | 3/64 | 3/64 | 2/64 | Pass |

The table shows the fresh harness pair on the shared image, using the local backend inside that image. Both runs completed the full three steps with the same seed, model and prompts. Earlier runs produced the same counts. The 75% pass threshold is 75 percentage points above the original run's lowest batch and 20.31 points below the repair's lowest batch. The recorded run starts scoring after the first wake.
