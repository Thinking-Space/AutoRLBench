# Tool observations are silently rounded and GRPO trains the agent to stop using the tool

Training Qwen3-1.7B with multi-turn GRPO and a Python-interpreter tool on GSM-Hard in verl, the agent loop rewrites every number in a tool observation to six significant digits. Tool-assisted answers to these large-number problems become wrong, and within about ten steps GRPO teaches the agent to stop calling the tool. Training reward still rises, but held-out accuracy averages 0.5668 and 0.5225 over steps 20-60 (seeds 42, 43) against 0.6625 and 0.6545 for the same recipe with verbatim observations, and the tool-call rate falls from 0.466 to 0.008.

| | |
| --- | --- |
| **Author** | pkulium |
| **Affiliation** | Iowa State University |
| **Type** | Environment |
| **Domain** | Tool-use RL |
| **Tags** | `tool-observation` `precision-loss` `tool-abandonment` `multi-turn` |
| **Framework** | verl 0.8.0 (GRPO, multi-turn tool agent loop, hermes tool calls), FSDP + colocated vLLM, `run_tool_gsm_hard.sh` |
| **Run time** | ~64 minutes |
| **Resources** | 1 × A100 (80 GB) |

## Difficulty

The task-written files are all correct:
- `tools/python_tool.py` runs the code in a subprocess and returns its output verbatim;
- `reward/numeric_reward.py` checks the final number against the gold (relative tolerance 1e-9,
  absolute 1e-4);
- the launch script is a plain verl tool-agent recipe.

The defect is two lines in the vendored verl agent loop. `ToolAgentLoop._call_tool` passes each
observation through `shorten_numeric_literals` (`verl/experimental/agent_loop/utils.py`), next to
the legitimate `max_tool_response_length` truncation. The agent's program prints `475501268356`;
the agent reads `4.75501e+11`. Nothing errors.

The evidence is only in the run artifacts. The transcripts show exact-integer programs answered
with exponent-form observations:
- `val_generations/0.jsonl`: 66 of 118 tool-using held-out responses carry an
  exponent-form observation, and 2 of them score.
- `rollouts/1.jsonl`: those rollouts score 0.065, against 0.543 without the tool.

GRPO acts on that gap:
- `num_turns/mean` goes 3.17 → 2.31 → 2.01 (steps 1 → 10 → 60);
- the held-out tool-call rate goes 0.466 → 0.008;
- the training score keeps rising (0.363 → 0.631) as the agent learns large-number arithmetic in
  its head.

The run reads as an agent that "learned it does not need the tool". To find the cause, an agent has
to:
1. notice that a correct Python `print` cannot produce `4.75501e+11` for an integer product;
2. connect that to the reward;
3. locate the rewriting outside the tool file.

| held-out accuracy (253 questions, greedy) | step 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 (shipped) | 0.3281 | 0.4348 | 0.5099 | 0.5692 | 0.5771 | 0.5810 | 0.5968 |
| defective, seed 43 | 0.3281 | 0.4111 | 0.4743 | 0.5020 | 0.5375 | 0.5336 | 0.5652 |
| control, seed 42 | 0.4585 | 0.5336 | 0.6561 | 0.6482 | 0.6561 | 0.6957 | 0.6561 |
| control, seed 43 | 0.4466 | 0.5020 | 0.6364 | 0.6680 | 0.6364 | 0.6403 | 0.6917 |

| held-out tool-call rate | step 0 | 10 | 20 | 30 | 40 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| defective, seed 42 | 0.466 | 0.067 | 0.036 | 0.024 | 0.016 | 0.008 | 0.008 |
| control, seed 42 | 0.458 | 0.510 | 0.751 | 0.767 | 0.798 | 0.925 | 0.889 |

Wrong diagnoses:
- **"the agent correctly learned the tool is unnecessary":** the controls keep using it and are
  more accurate.
- **The reward:** its tolerance is part of the task, and the control is scored by the same reward.
  "The reward rejects rounded answers" earns partial credit only.
- **The tool file:** it prints exact output.
- **The hermes parser, the dataset's noisy labels (both arms share them), the learning rate or the
  KL settings.**
- **"Healthy because accuracy rises".**

Horizon: 60 steps of 64 prompts × 8 samples (about 3.6 epochs of the 1,066 training problems), at
most 4 assistant turns, 1,024 response tokens. Greedy evaluation runs on all 253 held-out problems
at step 0 and every 10 steps.

From step 20 on, every control evaluation (lowest 0.6364) is above every defective one (highest
0.5968) across both seeds.

Issue-only: the closest seed-paired step-60 gap is 5.9 points on 253 problems, too thin
for a rerun threshold.

## Reference solution

- **issue**: a passing diagnosis must name `shorten_numeric_literals` in the agent loop and its
  effect, as one causal chain:
  1. tool observations are rounded to six significant digits;
  2. tool-assisted answers to large-number problems become wrong under the exact numeric reward;
  3. GRPO trains the agent to stop calling the tool.

  Naming only the tool abandonment, the precision loss without its location, or the reward's
  strictness earns partial credit. Fix for reference: remove the call
  (`evaluations/issue/reference/fix.diff`).

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root cause /
  evidence (0.2 / 0.6 / 0.2). Pass needs a verdict score of 1.0 and both other scores above 0.8.
- **Recorded comparison**: `evaluations/issue/reference/RESULTS.md`. Two seeds per arm, one run
  each, so the margins are not a variance estimate.

**How the runs were executed.**
- All runs used the declared image digest (`verlai/verl@sha256:4c43bbf1…`) on an A100-SXM4-80GB,
  with the workspace mounted at `/workspace`.
- They ran under Apptainer 1.4.0 (`apptainer exec --nv`, SIF converted from the same digest), not
  under Docker. The tool's subprocesses ran inside that container.
- `environment/prepare.sh` was run once in the same image, and its pinned sha256 checks passed.

**Not verified:** a Docker run and the harness `rerun` command.
