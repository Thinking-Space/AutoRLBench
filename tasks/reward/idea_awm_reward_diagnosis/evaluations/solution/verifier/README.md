# Solution verifier: replay classification

Deterministic, verifier-only grading — no training rerun, no LLM judge. The
agent's deliverable is the repaired
`data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl` in the materialized
workspace (plus `/workspace/report.md`, enforced by the harness report gate).

## Design

The grader replays 14 held-out episodes through the submitted rule file:

- **7 negatives** — the audited false-positive episodes from the issue-level
  curation (fixtures and final answers verbatim). These trajectories did not
  satisfy the requested task but receive full reward from the defective rules.
  A correct repair must judge them **incomplete**.
- **7 positives** — synthesized correct completions of the same seven tasks
  (the entity the task asked to create actually created, or the correct
  read-only answer). A correct repair must judge them **complete**.

Each case materializes its `initial_tables` / `final_tables` fixture dicts
into sqlite, then invokes the rule selected by `(scenario, task_idx)` — the
same lookup `awm.tools.find_scenario_entry` performs in training — as
`verify_task_completion(initial_db, final_db, final_answer)`. A case is
correct when the verdict matches `expect`. Score = correct / 14;
`required_score: 1.0`.

## Grading curve

| Submission | Score |
| --- | --- |
| unrepaired original rules | 7/14 = 0.5 (all negatives accepted) |
| reject-everything patch | 7/14 = 0.5 (all positives rejected) |
| partial repairs | proportional |
| correct repair of all 7 rules | 14/14 = 1.0 |

The curve is symmetric on purpose: a fix that only suppresses the bad episodes
scores the same 0.5 as no fix at all, so both directions of the repair are
required.

## Guards

- `required_entry_count: 10010` — the file must still hold every original
  entry; a hand-written subset or truncation is rejected with an error
  (exit 2) rather than scored.
- Malformed JSON anywhere in the file is fatal (the training bridge loads it
  wholesale); the grader surfaces the first bad line number.
- A rule that raises or times out (60 s per case) classifies that case as
  `error`, i.e. incorrect — a broken fix is a failed fix.
- Results are written atomically (`os.replace`, symlinks refused) to
  `experiment_metrics.json`, `metrics.json`, and `result.json`.

## Files

| Path | Purpose |
| --- | --- |
| `test.sh` | Entry point; requires `AUTORLBENCH_WORKSPACE` / `AUTORLBENCH_RESULTS`. |
| `tests/grade_submission.py` | Grader: load spec, validate the submitted rule file, replay, write metrics. |
| `resources/grader_spec.json` | Strict spec (schema v2, `score_mode: final`). |
| `resources/replay_cases.json` | The 14 replay episodes. |
| `resources/tools/replay_fixtures.py` | Self-contained materialization + replay runner (issue-level tool mirror). |

Validation evidence is in `../oracle/README.md`; the reference fix passes
14/14, the original defective file scores exactly 7/14 (see
`../oracle/validate_fix.py`).
