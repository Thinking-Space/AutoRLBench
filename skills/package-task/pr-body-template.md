# PR body template

The last step of packaging emits a PR body derived from the finished task `README.md`. The README stays the
single source of truth; the PR body is a derived view that gives maintainers the task at a glance. Do not save
it into the task directory — present it to the user in the conversation to paste into the PR.

The PR title is `[New Task] <task.toml title>`.

```markdown
`tasks/<domain>/<task_id>/`

<The README's one-sentence summary, verbatim.>

| | |
| --- | --- |
| **Author** | <name> |
| **Affiliation** | <institution> |
| **Type** | <Algorithm · Configuration · Reward · Data · Environment · Infrastructure> |
| **Domain** | <research area> |
| **Tags** | `<tag>` `<tag>` |
| **Framework** | <framework (algo), version pins> |
| **Run time** | <duration of the buggy run experiment> |
| **Resources** | <N × <GPU model> (<VRAM per card>), e.g. 8 × A800 (80 GB)> |

### Evidence

<2–4 sentences with the strongest measured numbers: both arms' outcomes, and with a solution tier the
threshold, its margins to both arms, and how many runs per side calibrated it. Copy the numbers from
the README — never restate them loosely.>
```

Rules:

- The table is copied row for row from the README, so a reviewer can diff the two at a glance.
- The summary sentence is the README's, verbatim.
- Only the Evidence section paraphrases (compactly); every number in it still comes from the README.
- The `.github/PULL_REQUEST_TEMPLATE.md` in the repo holds the same format for submitters who package
  without the skill.
