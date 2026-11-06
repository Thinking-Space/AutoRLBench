# Canonical issue oracle

Expected verdict: **unhealthy with respect to training/inference numerical consistency**. The historical run completes and learns; successful optimization does not establish agreement between the two forward computations.

## Single scored root cause

**The same synchronized nominal model weights produce significantly different token probabilities in the BF16 FSDP2 learner forward and vLLM generation forward.** The BF16 hybrid execution paths do not implement numerically equivalent conditional distributions. This is one numerical-consistency issue, not separate defects for the actor and rollout dtype flags.

Concrete implementation evidence, with paths relative to `workspace/`:

- `verl/verl/trainer/config/math_grpo.yaml` sets both `actor_rollout_ref.actor.dtype` and `actor_rollout_ref.rollout.dtype` to `bfloat16`. In `verl/verl/workers/fsdp_workers.py::_build_model_optimizer`, the actor dtype selects the FSDP2 `MixedPrecisionPolicy` parameter dtype; master parameter storage and gradient reduction remain FP32. `verl/verl/workers/actor/dp_actor.py::_forward_micro_batch` uses the actor dtype under `torch.autocast`. `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` passes the rollout dtype to vLLM. Learner batched/full-sequence scoring and autoregressive generation use different execution paths, so matching dtype names and nominal parameters do not guarantee matching floating-point results.
- `verl/verl/workers/sharding_manager/fsdp_vllm.py` synchronizes actor weights into vLLM. `verl/verl/trainer/ppo/ray_trainer.py::_compute_old_log_prob` recomputes learner `old_log_probs` and calls `scripts/run_support.py::parity_metrics` before `update_actor`. The comparison uses the same sampled response tokens and their prefixes, against the `rollout_log_probs` returned by generation. It is not the ordinary difference between a newly updated actor and its old rollout policy.

Expected behavior: closely agreeing token probabilities for these corresponding inputs and synchronized parameters. The paired precision experiment supports precision sensitivity, but does not isolate a particular CUDA kernel. Do not demand a more specific kernel-level attribution than the experiment establishes.

Full root-cause credit requires identifying the synchronized-weight BF16 FSDP2/vLLM probability disagreement, connected to the concrete configuration and comparison path. Equivalent code references and explanations count. A generic assertion that BF16 is inaccurate, without identifying the two probability computations, is incomplete.

In the context of these two forwards, **“the GPU computation paths differ, and BF16 has lower effective precision” is an acceptable full-credit root-cause explanation**: different execution/rounding paths together with limited BF16 precision produce different token probabilities despite synchronized nominal weights. These are two contributing aspects of the same issue, not two independent bugs. Do not demand the exact wording “synchronized nominal weights,” a named kernel, or the complete function inventory above when the diagnosis conveys this meaning. Score supporting workspace evidence separately from root-cause correctness.

## Visible evidence

`logs/run_001/metrics.jsonl` contains all 440 training iterations and the initial evaluation. `scripts/run_support.py::parity_metrics` defines the measurements using masked response tokens.

- Token-weighted mean absolute logprob difference: **0.0112136646 nat/token**, over **1,026,663,841** response tokens. The unweighted average over iterations is **0.0111310896 nat/token**. Neither absolute difference is a KL divergence. Report the measured scale rather than dismissing the mean as negligible.
- Maximum sampled-token absolute probability difference: **0.9999255938** at step **252**, computed as `abs(exp(old_log_prob) - exp(rollout_log_prob))`. This is a sampled-token discrepancy, not a vocabulary-wide distance.
- Maximum absolute logprob difference: **17.5484849 nat** at step **326**. The separate `parity_record.json` records actor logprob −18.8125019073 and rollout logprob −1.2640169859 for that event. This is a different event from the step-252 probability maximum.

High evidence credit requires concrete visible code/configuration evidence plus a correctly interpreted numerical probability or logprob discrepancy. The mean gap, a maximum gap, or another concrete discrepancy can support a complete diagnosis. Do not require every number, a particular collection of metrics, hidden comparator scores, or an analysis of sequence-MIS, ESS or rejection rates.
## Context, not additional grading requirements

- The formal BF16 run completes 440 iterations (1760 PPO minibatch optimizer updates per rank), with finite gradient audit records. AIME 2024 avg@32 improves from 0.228125 to 0.28645833; AIME 2025 improves from 0.19166667 to 0.24583333. A detailed evaluation analysis is optional; improvement does not invalidate the numerical-consistency diagnosis.

## Unsupported attributions

Do not accept missing weight synchronization, ordinary post-update policy drift, a label shift, an incorrect math grader, or TP8 versus SP1 as the demonstrated cause. The code and configuration do not support those explanations here. Claims of no learning, a collapsed/NaN formal run, or a fatal step-1 checkpoint error contradict the frozen evidence.

Do not require the agent to propose a repair. The evidence does not establish bitwise equality under FP16, identify a unique offending CUDA kernel, provide exact replay of a peak event, or prove a deployment-objective gap. These stronger claims are outside this task.
