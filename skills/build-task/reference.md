# build-task reference

The detail layer: SKILL.md carries the workflow and gates; this file carries the judgment
criteria, checklists, and risk lists.

## 1. Rejection criteria (P0)

Three necessary conditions — missing any one rejects the input:

1. **Root-cause level**: the claim points at concrete code, configuration, data
   processing, or reward logic, and carries a counterfactual ("fixing it restores
   healthy behavior"). A bare symptom ("reward went to zero") is not a claim; it is a
   diagnosis waiting to happen.
2. **Falsifiable**: an expected signal can be stated — metric, direction, magnitude, and
   roughly when it appears. An unfalsifiable claim cannot gate a pre-experiment.
3. **Training-run granularity**: the effect must be observable inside a complete RL
   training/rollout loop. A library-level bug with no behavioral consequence in the loop
   is unit-test material, not a task.

Three funnels (is the setting instantiable — all cost zero GPU time):

- **Image funnel**: the framework must fit one of the fixed shared images in
  [ENVIRONMENTS.md](../../ENVIRONMENTS.md). If it does not, port to a supported stack or
  raise the image-list question with the maintainers.
- **Model funnel**: check the cheapest kills first — does the rollout engine support the
  architecture, does a training recipe exist, do tokenizer/tool-call parser bindings
  match, does the tokenizer's `transformers` class exist? Then: can the assets be
  obtained at pinned revisions, and does the failure mode **require a minimum scale to
  manifest**? Memory-pressure and distributed-dispatch failures need scale; parser
  mismatches and retry-fanout failures show at any scale. Choosing against this either
  burns budget for nothing or never shows the defect.
- **Recipe funnel**: deviations from the framework's default recipe or the paper's
  official reproduction script need a written reason (SPEC field 6). An unjustified
  customization plants a distractor: readers cannot tell whether the effect comes from
  the defect or the setting.

Rejection note and destinations:

| Problem with the input | Destination |
| --- | --- |
| Symptom without attribution | upstream diagnosis (not this skill's work; return when a root cause exists) |
| Setting fits no shared image | back to research / port the stack / ask maintainers |
| No basis for a healthy baseline | add literature or a pre-experiment, else drop |
| Bug has no consequence inside the training loop | drop, or keep as unit-test material |
| B3 pure reasoning | release only with user confirmation; strictest pre-experiment gate |

## 2. SPEC field guide

The template is [templates/spec-template.md](templates/spec-template.md); the judgment
criteria per field:

| # | Field | Key criterion |
| --- | --- | --- |
| 1 | Fault claim | one sentence containing setting S, defect D, observable O, healthy H; the counterfactual holds |
| 2 | Source and level | B1 (cited incident/observation) / B2 (mechanism class) / B3 (reasoning) / A; with links and numbers |
| 3 | Framework, version, image row | digest-pinned; what the image lacks goes to `prepare.sh` |
| 4 | Models | each role (policy / reward / judge) passes the three funnels; conclusions recorded |
| 5 | Datasets | train prompts **and** a held-out eval the failure is measurable on |
| 6 | Recipe | default-by-the-book, or the customization and its reason |
| 7 | Defect carrier | where the bug lives (flag / patch / data processing); the expected two-arm diff shape |
| 8 | Expected signal (pre-registered) | metric key, direction, magnitude, step of appearance. **Written and frozen before any full run in prospective mode** |
| 9 | Healthy baseline expectation | the range the fixed arm should land in; catches both-arms-broken |
| 10 | Distractor list | plausible-but-wrong diagnoses with exclusion reasons |
| 11 | Budget | debug config, full config (both arms), GPU count, wall-clock cap |

Common attribution trap: "training did not improve" is not the root cause when the
defective arm *also* improved — the claim is "lower than control". Both arms improving
does not mean there is no defect.

## 3. Evidence spectrum details

- A-both-arms still has work: the SPEC is written **as a faithful record** of why the
  original runs were configured the way they were. If the original run was not executed
  inside a shared image, it must be re-run there — at that point a retrospective build
  partially becomes a constructed one.
- Retrospective mode records; prospective mode chooses-and-justifies. Same template,
  different epistemic status.
- **Family discount**: a variant of an existing task family inherits most SPEC fields
  from its sibling; only fields 1, 7, and 8 change, and inherited fields are labeled.
- The level determines exactly two things: how much evidence must be constructed, and
  how strict the pre-experiment gate is. It never changes the logical structure.

## 4. Pre-experiment design (P5 step 3, prospective mode only)

- Cost target: roughly one tenth of a full run — fewer steps, less data, or a smaller
  model, **without shrinking past the failure mode's minimum manifestation scale** (see
  the model funnel).
- The judgment is SPEC field 8 answered item by item: does the expected signal appear,
  in the right direction, with enough magnitude to separate the arms (a thin margin
  counts)?
- A miss → revise the setting and re-gate, or drop. **Dropping is a pipeline output**:
  the GPU-hours a pre-experiment saves are worth as much as a task.
- Failure patterns to expect (both observed in practice): the suspected differential
  effect turns out to be a common-mode improvement present in both arms; or a replication
  shows the claimed gap is within run-to-run noise. Both are cheap to discover at
  one-tenth cost and expensive at full cost.

## 5. Hand-off checklist (P7)

The phase gates of this skill are derived by inverting validate-task's checklist, so a
build that passes its own gates lands reviewable:

| validate-task section | Responsible phase | Concrete output |
| --- | --- | --- |
| §2 complete RL experiment | P0 + P5 | SPEC invariants, debug-loop criterion |
| §3 buggy-run evidence | P5 + P6 | full-run artifacts, frozen collection, logs with identity information |
| §4 assets reproducible | P1 + P2 + P4 | image pin, asset pins, durable layout, idempotent prepare |
| §4a rerun startup | P5 | prepare can launch the experiment |
| §5 root cause and tier | P0 + P5 | SPEC fields 1/8/10, two-arm reading |
| §6 verifier / scoring | package-task | not this skill |
| §7 licensing / hygiene | P6 + package-task | sanitization record |

Hand-off checklist (= package-task's required inputs; a missing item stops the hand-off —
ask, do not guess):

- [ ] Source material, form 2: `buggy_workspace/` + `fixed_workspace/` (or declare form 1 / form 3)
- [ ] Launch script (path relative to `workspace/`), positional args, env vars, and in-container paths for model, train set, and validation set
- [ ] Logs of the failed run (including offline experiment-tracking files, if the run used them)
- [ ] Environment: the shared-image row, confirmation the buggy run executed inside it, and the GPU count to declare
- [ ] Every asset: source, 40-char revision or SHA-256 + byte size, and destination
- [ ] If a solution tier is intended: metric key, one log sample, `final_step`, `required_final_score`, and how many runs per side calibrated the threshold

## 6. Prospective-mode risks (P5 / P6 watchlist)

1. **Implicit p-hacking**: building both arms yourself invites picking the seed and
   hyperparameters that maximize the effect. The defenses are the pre-registered signal
   (SPEC field 8) and the arm-purity diff gate — both **before** the full runs.
2. **Both arms broken**: the fixed arm must also be checked against SPEC field 9; if it
   misses the healthy baseline, both arms are void — no grading on a curve.
3. **Upstream already fixed**: when the defect is a real upstream bug, the pinned
   revision must be the buggy one (check the fix commit's relationship to it first).
4. **Design leakage**: the SPEC, research notes, and survey material stay out of
   anything publishable; they are the answer, the same way `fixed_workspace/` is.
5. **Budget overrun**: a prospective build has no sunk-cost guardrail. Respect SPEC
   field 11 — descale or drop rather than push through.

## 7. Resumption and drift

- The state file's **external handles** (environment IDs, storage paths, asset
  locations, exposed URLs) are re-verified before use on every resumption.
- Mixed runs in a results directory are filtered to the target run by timestamp before
  any evidence work; failed imports' output directories are archived so retry loops do
  not fill storage with empty trials.
- Anything the environment rebuilds on restart (virtualenvs, checkouts, caches) is
  rebuilt by the idempotent `prepare.sh`, not by hand; durable artifacts were never in
  the ephemeral layer to begin with.
