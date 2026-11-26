# Repository-repair reward pipeline silently mislabels rollouts

An agentic RL run on repository-repair tasks rewards rollouts by running the repositories' own
tests on patched checkouts, but four pipeline defects make good patches unrewardable, let rollouts
rewrite the test surface, mis-match node ids, and turn undecidable runs into confident zeros.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Environment |
| **Domain** | SWE |
| **Tags** | `reward-pipeline` `test-execution` `log-parsing` `reward-hacking` `data-quality` |
| **Framework** | none (the graded artefact is the reward service; the rollouts it grades are staged) |
| **Run time** | CPU only; prepare ~100 s (reproducing the staged grading pass of 194 records locally takes ~4.5 min on 8 workers) |
| **Resources** | CPU only, no GPU, no network after prepare |

## Difficulty

Nothing in this workspace crashes. The pipeline grades every record, writes a reward for every
record, and its unit tests are green -- they feed the parser logs that already contain the
section the real grading command never asks pytest to print. The repositories are healthy:
`python3 -m pytest -q` in any snapshot under `corpus/repos/` is green in seconds in the
`.venvs/<repo>` environment that was installed for it. The only external symptom is a flat
training reward, and the archived pass behind it looks like a corpus of uniformly bad rollouts.

Four defects sit in four different stages, and they mask each other:

| Stage | Defect | Effect on the reward |
| --- | --- | --- |
| `command.py` | the grading command omits `-rA`, while the parser only reads the `short test summary info` section | a run in which everything passes parses to nothing → reward 0 |
| `reset.py` | restores `test_*.py` / `*_test.py` only, per file, with `2>/dev/null` and no status check | a rollout's `conftest.py` / `pytest.ini` report hook decides its own result |
| `log_parse.py` + `match.py` | node ids lose their file path, and an absent id is resolved by substring fallback | same-named tests in different files merge; unreported ids get another test's status |
| `reward.py` | no label for "the run could not decide" | timeouts and never-reported tests enter the advantage as zeros |

Fixing the first defect alone turns the pipeline from "rewards nothing" into "rewards the
cheating rollouts", which is worse: measured on the staged corpus, 0 → 109 rewarded records, with
68 wrong `resolved` labels in the calibration. Those false positives include every one of the 30
`conftest.py`-hook rollouts, 25 of the 30 `pytest.ini` plugin rollouts, and records whose run never
reported a required test. That is the trap in
this task: the logged, obvious symptom and the thing that actually makes the reward trustworthy
are different defects, and the corpus contains the rollouts that punish stopping early.

| Pipeline | rewarded | agreement with the reference labels | wrong `resolved` | `unscorable` recall |
| --- | ---: | ---: | ---: | ---: |
| as shipped | 0 / 0 | 0.680 / 0.614 | 0 / 0 | 0.00 / 0.00 |
| grading command repaired only | 109 / 87 | 0.608 / 0.665 | 68 / 46 | 0.00 / 0.00 |
| command + test-surface restoration | 54 / 54 | 0.892 / 0.873 | 13 / 13 | 0.00 / 0.00 |
| reference (every stage) | 41 / 41 | 1.000 / 1.000 | 0 / 0 | 1.00 / 1.00 |

Each cell is *staged corpus (194 records) / held-out corpus (158 records)*, measured at
submission time on corpora that share no instance. The third row is the trap's far side: a
pipeline that asks pytest for the right log and restores the test suite still rewards 13
records it cannot decide, because the matcher answers a required id the suite never reports
with another test's result. Node-id matching and the `unscorable` outcome are one step, not
two: with the substring fallback in place a missing id is never *observed* to be missing, so
the third label can never be produced.

Shortcutting the test run is no help either. The corpus pairs each rollout archetype
with its mirror image: a patch that fixes the bug *and* carries a report hook (`resolved`)
against a patch that carries only the hook (`unresolved`), and a patch that touches only source
files and still fails (`unresolved`) against one that fixes it (`resolved`). A grader that
labels submissions by what they touch gets all four groups wrong.

Plausible but wrong diagnoses: the repositories or their pinned environments are broken (their
suites are green); the injected bugs do not apply (they do, and `git status` in a scratch
checkout shows them); the rollout patches are bad (30 of them are the exact inverse of the
injected bug); the unit tests are missing a case (they pass, on logs the real command cannot
produce).

The corpus is public data: six permissively licensed Python repositories, synthetic injected
bugs from the public SWE-smith dataset, and 352 rollout patches against those snapshots -- 11
recorded from a training run (6 of which genuinely fix their instance), the rest staged for this
task by patching real checkouts (see `workspace/THIRD_PARTY_NOTICES.md`).

Two properties of the corpus are deliberate and are part of the defect being graded, not
accidents of packaging. 60 of the 194 staged records change no source file at all -- they add a
`conftest.py` report hook or register a plugin through `pytest.ini`; that is what the original
run's rollouts learned to do once the reward could be moved at all. And 8 staged / 4 held-out
instances carry one required node id that the pinned environment never produces, reproducing the
undecidable records of the bank this corpus is modelled on (in the original audit, 67 of 293
records were of that kind). Both are visible in the workspace; neither is labelled there.

## Reference solution

- **issue**: name the four defects with their mechanism and direction -- the command/parser
  contract that makes every reward 0, the test-surface restoration that lets a rollout grade
  itself, the node-id matching that answers with another test's result, and the missing
  "undecidable" outcome -- and support each with visible source, archived logs, or a
  local reproduction. The archived corpus counts are examples, not required recall.
  A diagnosis that stops at the missing `-rA` has found the defect that explains the
  flat reward, not the one that makes the reward wrong.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root
  cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires verdict = 1,
  root cause > 0.8, and evidence > 0.8, not merely a total ≥ 0.8. The oracle
  carries four essential root causes; concrete support for each is required,
  but exact counts of patches or undecidable records are not. Naming only the
  missing summary section cannot pass.
  `environment/check_evidence.py` re-derives every number quoted here from the shipped
  artefacts and scans the task for private markers.
