# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the defective workspace and the
recorded control (same code, data, seed, budget and image; the control lacks the two-line call
and the helper in `verl/experimental/agent_loop/`; nothing else differs).

Expected verdict: **unhealthy**.

## Root cause

- `verl/experimental/agent_loop/tool_agent_loop.py::ToolAgentLoop._call_tool` calling
  `verl/experimental/agent_loop/utils.py::shorten_numeric_literals`
  - kind: `symbol`
  - **observed:**
    - Every tool observation goes through `shorten_numeric_literals` before it is appended to the
      conversation. The function sits just before the existing `max_tool_response_length`
      truncation.
    - It rewrites every number with more than six digits as a `%g` literal with six significant
      digits. `print(689566 * 689566)` prints `475501268356`, but the agent sees `4.75501e+11`.
      `4561195.2` becomes `4.5612e+06`.
    - GSM-Hard answers are mostly large exact numbers. When the agent computes the answer with the
      tool, it reads a rounded value and writes a wrong final answer, for example `4561200` for
      the gold `4561195.2`. Sometimes it writes the exponent form instead.
    - `reward/numeric_reward.py` is exact for large integers (relative tolerance 1e-9) and scores
      either form 0, even when the agent's program was correct.
    - GRPO therefore sees tool-assisted rollouts scoring far below rollouts that skip the tool. It
      trains the policy to stop calling the tool.
  - **expected:** observations reach the agent exactly as the program printed them, as in the
    control. There, tool use pays off: the tool-call rate rises and held-out accuracy is higher.
  - `tools/python_tool.py` itself is correct and identical in both arms.

A complete attribution names the observation rewriting in the agent loop and its causal role, as
one chain:
1. tool outputs are rounded to six significant digits;
2. tool-assisted answers to large-number problems become wrong;
3. GRPO learns to avoid the tool.

Do not require additional causes, and do not require a statement about final accuracy relative to
other runs.

**The `shorten_numeric_literals` rewriting is required.** Without locating it, the following earn
at most partial root-cause credit:
- "the policy stopped using the tool";
- "tool outputs are in scientific notation / lose precision";
- "the reward is too strict for rounded answers".

These earn no root-cause credit:
- the tool implementation in `tools/python_tool.py`, which prints exact output;
- the hermes parser;
- the dataset;
- the learning rate or KL settings;
- calling the run healthy because training score and held-out accuracy rise.

## Observable impact

Every item here is derivable from the shipped workspace (`results/tool_gsm_hard_seed42` plus the
code and configuration). "Exponent-form observation" means a `<tool_response>` block containing a
number that matches `\d\.?\d*e[+-]\d+`.

- **Observations are rounded.** The programs print exact integers, but the observations are in
  exponent form.
  - `val_generations/0.jsonl`: 118 of 253 held-out responses call the tool;
    66 of them contain an exponent-form observation, and 2 of those score.
    Responses without the tool score 0.519.
  - `rollouts/1.jsonl`: 234 of 512 rollouts call the tool and 123 contain
    exponent-form observations. Those score 0.065, against 0.543 for rollouts
    without the tool.
- **Tool use collapses.** `num_turns/mean` goes 3.17 → 2.31 → 2.01 at steps 1 → 10 → 60, and 2 means no
  tool call. The held-out share of responses with `<tool_call>` goes 0.466 (step 0) →
  0.067 (10) → 0.036 (20) → 0.008 (60). `rollouts/60.jsonl` contains
  2 tool-using rollouts out of 512.
- **The run still looks healthy.** `critic/score/mean` goes 0.363 → 0.631, and logged
  `val-core/gsm_hard/acc/mean@1` rises from 0.3281 (step 0) to 0.5968 (step 60).
- **Nothing errors.** The tool runs, the calls parse, and the observations are well formed.

## Trigger conditions

- Answers that are copied from tool observations, numeric answers with more than six significant
  digits, and an exact numeric reward. On small-number problems the rewriting changes nothing.
- On-policy GRPO, which turns the reward gap between tool-assisted and tool-free rollouts into a
  policy that avoids the tool within about ten steps.
