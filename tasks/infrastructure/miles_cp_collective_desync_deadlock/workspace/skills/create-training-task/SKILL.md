---
name: create-training-task
description: Create new Miles RL training tasks, including data selection, reward function design, single-turn QA reuse, multi-turn agent rollout setup, MiniCPM5 MoE routed-expert handling, and recommended IcePop/R3/dynamic-sampling settings.
---

# Create Training Task

## Goal

Use this skill when adding a new training task in Miles. Always decide these first:

- What data to train on, and whether it is single-turn QA or multi-turn/agent data.
- What reward function is required, and whether existing reward code can be reused.
- Which launch script is closest to the target model and task.
- Whether the model is MoE, especially MiniCPM5 `16a3b`, where routed expert behavior needs special handling.

## Missing Inputs

If the user request is underspecified, ask for the missing inputs before creating files or changing training scripts. Required inputs:

- Task type: QA/single-turn, long-context single-turn, multi-turn agent/tool task, or other.
- Model and checkpoint: model family/size, HF checkpoint path, Megatron `--load` path, and target save path.
- Data: train data file
- Reward: existing reward type, custom reward behavior, or scoring contract.
- For agent tasks: tool specs path, tool execution function, parser format, max turns, and sandbox/dependency requirements.

When reasonable defaults exist in the closest script, propose them to the user and state which inputs still need confirmation. Do not invent dataset paths, checkpoint paths, or reward semantics.

## Pick a Template

When creating a new task, first look for the closest existing script under `scripts/minicpm5` and adapt it instead of starting from scratch.

For MiniCPM5:

- Single-turn short QA/math: start from `scripts/minicpm5/16a3b/short_rl/run-minicpm5-moe-16a3b-short-math-8k-full-async.sh`.
- Long single-turn: start from `scripts/minicpm5/16a3b/long_rl`.
- Multi-turn agent/tool task: start from `scripts/minicpm5/16a3b/agent_rl/run-minicpm5-moe-16a3b-retool-v2-math-128k-full-async.sh`.
- For `1b` or `2_6b`, use the matching `scripts/minicpm5/<size>` directory.

Copy the closest script, then change `TASK_TAG`, `TRAIN_FILE`, `TEST_FILE`, prompt/label/tool keys, reward settings, rollout lengths, and checkpoint paths.

## Data Decisions

Check the dataset schema before editing code:

- QA/single-turn usually needs `prompt` and `label`.
- Agent/multi-turn usually needs `prompt`, `label`, and `tools`.
- Ensure `--input-key`, `--label-key`, and `--tool-key` match the JSONL fields.
- Tune `ROLLOUT_MAX_PROMPT_LEN`, `ROLLOUT_MAX_RESPONSE_LEN`, `ROLLOUT_MAX_CONTEXT_LEN`, and `SGLANG_CONTEXT_LEN` to the actual task length.
- Keep evaluation data separate and named in `TEST_FILE` as `name path name path`.

## QA / Single-Turn Tasks

For QA tasks, reuse the existing single-turn implementation whenever possible:

- Keep the fully-async rollout path: `--rollout-function-path fully_async_rollout.generate_rollout_fully_async`.
- Use existing reward types such as `--rm-type math` if they fit.
- If only reward logic changes, implement a new reward function rather than a new rollout.
- Reward code usually belongs in `miles/rollout/rm_hub`.
- If using a custom reward, wire it with `--custom-rm-path package.module.function` and set `--reward-key score` if the function returns a score field.

Prefer small focused reward functions: parse answer, compare to label, return stable numeric rewards, and handle malformed outputs deliberately.

## Multi-Turn / Agent Tasks

For simple stateless multi-turn tasks, reuse the existing multi-turn generate path:

```bash
--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
--generate-tool-specs-path <tool_specs_module_or_func>
--generate-execute-tool-function-path <execute_tool_module_or_func>
--generate-tool-call-parser minicpm4_xml
--generate-max-turns <N>
--log-multi-turn
```

Stateless means each turn can be derived from the current conversation, tool specs, and tool result without persistent environment/session state outside the sample.

For stateful multi-turn tasks, create a new generate implementation instead of forcing the task into the existing generic multi-turn flow. Stateful means the task has persistent environment state, session handles, external resources, hidden simulator state, long-lived tool state, or reset/step semantics that must be controlled across turns. Put the new implementation under `miles/rollout/generate_hub` or a task-specific package, then wire it with `--custom-generate-function-path`.

Agent tasks usually need:

- Dataset rows with tool definitions or a tool key.
- Tool specs and execution functions that are deterministic and sandboxed.
- A reward function that scores final task success, not just format.
- Tests or dry-runs for tool call parsing, tool execution, truncation, and multi-turn merge behavior.

Use `miles/rollout/generate_hub/multi_turn.py` and `miles/rollout/generate_hub/agentic_tool_call.py` as the primary implementation references.

## MiniCPM5 16A3B MoE Handling

For MiniCPM5 `16a3b`, keep the MoE-specific training/inference settings aligned with the existing scripts:

- Source `scripts/minicpm5/16a3b/minicpm5_moe_16a3b_rope.sh`.
- Use `--expert-model-parallel-size`, `--expert-tensor-parallel-size`, TP, CP, and rollout GPU settings from the closest existing script.
- Keep `--moe-router-dtype fp32`.
- Keep SGLang MoE guards used by the existing scripts, such as `--sglang-disable-shared-experts-fusion` and `--sglang-disable-overlap-schedule`, unless there is evidence they are no longer needed.
- Enable routed-expert replay/R3-style handling for train-infer consistency:
  - `--use-rollout-routing-replay`
  - `--use-miles-router`
  - `--use-tis`
  - `--custom-tis-function-path miles.backends.training_utils.loss.icepop_function`

Do not remove routed expert handling when converting a single-turn template into an agent template.

## Recommended Algorithm Settings

Default recommendation:

- Use IcePop/TIS: `--use-tis --custom-tis-function-path miles.backends.training_utils.loss.icepop_function`.
- If the model is MoE, enable R3/routing replay via `--use-rollout-routing-replay` and `--use-miles-router`.
- Enable dynamic sampling to improve efficiency and stability:
  - `--over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))` for short tasks.
  - Larger oversampling such as `* 4` for long/agent tasks if resources allow.
  - `--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.
- Start with GRPO settings from existing scripts unless the task explicitly needs PPO or another estimator.

## Validation Checklist

Before handing off:

- The launch script sources the correct model args file.
- `PYTHONPATH` includes `Megatron-LM`, `mbridge`, repo root, SGLang python path, and any examples package used by tools/rewards.
- `TRAIN_FILE` and `TEST_FILE` exist and match the configured keys.
- Reward function path imports successfully.
- Multi-turn tool specs, parser, and execute function import successfully.
- MoE scripts preserve routing replay/R3/IcePop-related args.
- Run a small dry-run if feasible by reducing rollout count, batch size, and save interval.
