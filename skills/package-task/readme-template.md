# Task README template

Every task's `README.md` follows this template, modeled on terminal-bench-science. The reader is a reviewer, not the agent:
the README may state the root cause and threshold, but must not be mounted into the agent stage.

Style:

- **Clear at a glance.** The sentence under the title says "what is being trained and what goes wrong" — it carries the
  model, algorithm, and dataset (there is no table row for them), matching the sentence on the website task card.
- **The identity table has exactly eight rows**: Author, Affiliation, Type, Domain, Tags, Framework, Run time, Resources.
  `Type` is the task's domain directory, capitalized (`Algorithm`, `Configuration`, `Reward`, `Data`, `Environment`,
  `Infrastructure`). `Domain` is the research area (`Math RL`, `Code RL`, `Search RL`, `Tool-use RL`, …). `Tags` are
  kebab-case failure-mode descriptors, not framework names. `Run time` is the duration of the **buggy run** experiment
  (approximate forms like `~45 minutes` or `≤ 3 hours` are fine). `Resources` is a GPU count, e.g. `8 GPU`.
- **Only three sections**: Difficulty, Reference solution, Verification. The section *titles* are required; their content
  is free-form. The only structural requirement is that Reference solution and Verification each carry at least
  `- **issue**` and `- **solution**` bullets (an issue-only task omits `- **solution**`). Maintainer notes
  (migration history, asset shas, sanitization records) go in `prepare.sh` comments or `source_manifest.json`, not the README.
- **Always use numbers**, never "dropped significantly".

```markdown
# <Title, same as task.toml title>

<One sentence: training <model> with <algo> on <dataset>. <What goes wrong>.>

| | |
| --- | --- |
| **Author** | <name> |
| **Affiliation** | <institution> |
| **Type** | <Algorithm · Configuration · Reward · Data · Environment · Infrastructure> |
| **Domain** | <research area, e.g. Math RL · Code RL> |
| **Tags** | `<tag>` `<tag>` |
| **Framework** | <framework (algo), version pins> |
| **Run time** | <duration of the buggy run experiment> |
| **Resources** | <N GPU> |

## Difficulty

<Free-form: where the defect is, why nothing fails loudly, the strongest evidence
against the wrong diagnosis. Optional: a small table of the measured curve.>

## Reference solution

- **issue**: <what a passing diagnosis must name; which attributions must fail.>
- **solution**: <the minimal patch and why it is sufficient.>

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: the agent submits `report.md` and does not train; the harness reruns the
  experiment per `run_spec.json` and grades only that run. <metric key, step, threshold.
  Margins to both arms, whether it clears the control's step 0, and how many runs per side
  it was calibrated on.>
```

For issue-only tasks, delete the `solution` bullet from both Reference solution and Verification; keep the `- **issue**`
bullets. Model, dataset, environment image, and run counts are no longer table rows — fold anything a reviewer needs
into the summary sentence, `Framework`, or the section bodies.
