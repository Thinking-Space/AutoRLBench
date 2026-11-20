---
name: analyze-training-crash
description: Analyze Miles training crashes and instability using logs, metrics, train-inference mismatch signals, IcePop/R3 configuration, entropy, grad_norm, response_length, truncated_ratio, reward, dynamic sampling, and SGLang/Ray/Megatron failure evidence.
---

# Analyze Training Crash

## Triage Goal

Use this skill when a Miles training task crashes, diverges, hangs, OOMs, or becomes unstable. Start from evidence, then decide whether the issue is:

- Infrastructure/runtime: Ray, SGLang, NCCL, CUDA, process death, OOM.
- Configuration mismatch: model args, checkpoint, TP/CP/EP, context length, tokenizer/chat template.
- Train-inference mismatch: rollout distribution differs too much from actor training distribution.
- RL instability: reward variance, entropy collapse/spike, grad norm spike, excessive truncation, bad dynamic sampling.
- Task implementation bug: reward, multi-turn parser, tool execution, dataset schema, or prompt formatting.

## Missing Inputs

If the user only says the run crashed or is unstable, ask for the missing evidence before diagnosing. Required inputs:

- Launch script or exact command, including runtime env overrides.
- Error logs around the first failure, not only the final stack trace.
- Optional: screenshots of TensorBoard/WandB or other metric dashboards around the failure.

If some inputs are unavailable, ask for the smallest useful subset first: launch script, first error log, and metric snapshot near the failing step. Do not claim a root cause without at least one concrete log, metric, or config signal.

## Gather Evidence

Collect these before changing code:

- Full launch script and runtime env overrides.
- Ray job logs and actor/rollout logs.
- SGLang logs around the first error.
- TensorBoard/WandB metrics around the first abnormal step.
- Checkpoint iteration, HF checkpoint path, and Megatron `--load` path.
- Recent changes to reward, generate function, tool execution, model args, or data.

If there are many logs, search first for `Traceback`, `ERROR`, `OOM`, `CUDA out of memory`, `NCCL`, `NaN`, `inf`, `timeout`, `Killed`, `router`, `routing`, `truncated`, and `reward`.

## Train-Inference Mismatch

Check whether rollout and training behavior diverged too much:

- Was IcePop/TIS enabled?
  - `--use-tis`
  - `--custom-tis-function-path miles.backends.training_utils.loss.icepop_function`
  - reasonable `--tis-clip-low` and `--tis-clip`
- For MoE, was R3/routing replay enabled?
  - `--use-rollout-routing-replay`
  - `--use-miles-router`
- Did routed expert settings change between rollout and actor?
- Did SGLang use a different tokenizer, chat template, context length, dtype, or MoE fusion behavior from training?
- Did the checkpoint loaded by actor match the HF checkpoint used by rollout?

For MiniCPM5 `16a3b`, suspect train-inference mismatch early if routing replay/R3 is missing or SGLang MoE settings differ from the existing scripts.

## Metric Checks

Inspect metrics around the first bad step, not only the final crash:

- `entropy`: collapse can indicate over-confident policy or bad reward pressure; sudden spike can indicate distribution shift.
- `grad_norm`: spikes often precede NaN/inf, optimizer instability, bad batches, or too-large effective batch changes.
- `response_length`: sudden growth may cause memory pressure, truncation, or reward hacking.
- `truncated_ratio`: high or rising values mean `ROLLOUT_MAX_RESPONSE_LEN`/context length is too small or the policy is running away.
- reward mean/std/nonzero ratio: zero std breaks GRPO usefulness; high variance may destabilize training.
- KL/TIS/ratio metrics: clipping saturation can indicate train-inference mismatch or stale logprobs.
- tokens per GPU and dynamic batch metrics: spikes can explain OOMs or slow/hung steps.

Do not treat the crash stack alone as the root cause if metrics degraded earlier.

## Logs to Interpret

Common patterns:

- CUDA OOM during rollout: reduce `SGLANG_MAX_TOTAL_TOKENS`, `SGLANG_MAX_RUNNING_REQUESTS`, context length, response length, or memory fraction.
- CUDA OOM during training/logprobs: reduce `MAX_TOKENS_PER_GPU`, `LOGPROBS_MAX_TOKENS_PER_GPU`, batch size, or increase recompute.
- NCCL timeout/hang: check dead ranks, OOM before timeout, TP/CP/EP mismatch, and per-rank divergence.
- Reward import failure: verify `--custom-rm-path`, `PYTHONPATH`, and function signature.
- Multi-turn parser/tool errors: verify `--generate-tool-call-parser`, tool specs, tool key, sandbox dependencies, and truncation.
- Checkpoint load failure: verify model args, `--hf-checkpoint`, `--load`, `latest_checkpointed_iteration.txt`, and iteration directory.

## Recommended Stabilizing Configuration

When instability is not yet diagnosed, compare the run to these defaults:

- Use IcePop/TIS for train-inference correction.
- For MoE, enable R3/routing replay through Miles Router.
- Enable dynamic sampling:
  - `--over-sampling-batch-size`
  - `--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
- Keep GRPO conservative at first: low/zero entropy coef, clipped ratios, and existing script defaults.
- For long or agent tasks, lower rollout concurrency before changing algorithmic behavior.

These are recommended defaults, not proof of root cause. Confirm with metrics and logs.

## Analysis Output

When reporting back, use this structure:

1. Symptom: what failed and at which step/component.
2. Most likely root cause: one concise hypothesis backed by evidence.
3. Supporting signals: metrics/log lines/config comparisons.
4. Fix: smallest change to test first.
5. Follow-up checks: metrics to watch in the next run.

Avoid vague conclusions such as "training is unstable" without naming the first abnormal metric or log evidence.
