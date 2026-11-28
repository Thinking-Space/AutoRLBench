# Slime MoE router bias corrupted by BF16 loading

Training a tiny 8-expert MoE policy with grouped policy gradient on a deterministic two-action reproduction silently corrupts FP32 router correction biases when Slime's Bridge load copies them into a BF16 destination.

| | |
| --- | --- |
| **Author** | Jiankai Sun |
| **Affiliation** | Independent Researcher |
| **Type** | Infrastructure |
| **Domain** | Reinforcement Learning |
| **Tags** | `moe-router` `bf16-corruption` `checkpoint-loading` `silent-corruption` `controlled-reproduction` |
| **Framework** | PyTorch 2.11 controlled reproduction of the Slime loader |
| **Run time** | < 1 minute |
| **Resources** | CPU-only |

## Difficulty

The load completes without an exception: converting the model to BF16 changes the existing router destination buffer, and the subsequent FP32 source copy rounds eight values near 40 (spread 0.07) into one BF16 value. Casting that already-corrupted buffer back to FP32 later cannot recover its ordering. The defective run selects expert 0 and ends at 0.02040 correct-action probability; the control selects expert 7 and ends at 0.95893.

The packaged defective workspace and hidden control evidence are the official shared-image A/B from Slurm job 23051184. Both arms ran in the task's pinned `slimerl/slime:v0.3.2-cu129` digest; the imported SQSH had SHA-256 `9a3c9ab63f30a5b524744dc56173fe3e37d99e9a1c2dbafde307733e3ef67819`.

The hidden solution oracle retains the submission script, image-import log, Slurm accounting, complete repaired loader, and hashes of both source sides and all metric artifacts. The two arms share byte-identical `train.py`; only the loader repair differs.

Historical production-scale MoE observations establish the origin of the failure mechanism. The executable and graded task is a public, deterministic PyTorch grouped-policy-gradient reproduction of the Slime loader ordering, not a full Slime training stack. Unrelated historical timezone, router-update-rate, auxiliary-loss, batching, OOM, and other fixes are outside scope.

## Reference solution

- **issue**: a passing diagnosis must identify `checkpoint.py::load_hf_weights`: it copies FP32 correction bias into an existing BF16 router buffer after model conversion, collapsing router ordering. Claims about timezone, update rate, auxiliary loss, batching, OOM, or a general failure to learn are not the root cause.
- **solution**: before the copy, traverse each model chunk and call `_maintain_float32_expert_bias()` on modules that provide it. This pre-load loop is the only primary repair.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2, with the verifier's component pass conditions.
- **solution**: the agent submits `report.md` and does not train; the harness reruns the 12-step experiment per `run_spec.json`. At step 12, `eval/correct_action_probability` must be at least 0.955 (0.00393 below the control and 0.93460 above the defective arm), which is above the control's step-0 value 0.95257. The same row must report FP32 router bias, 8 unique values, matching source and router hashes, and selected expert 7. Calibration uses one deterministic run per side; margins, not variance, are reported, and no steps beyond 12 were verified.
