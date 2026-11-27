---
name: build-task
description: >-
  Produces AutoRLBench raw material from a fault claim: either an existing
  buggy run (retrospective mode) or a researched failure idea (prospective
  mode) is driven through design, shared-image selection, asset acquisition,
  buggy/fixed two-arm execution, evidence collection, and sanitization, until
  the result matches the raw-task format in CONTRIBUTING.md and is ready for
  the package-task skill. Use when asked to build, construct, or prepare an
  AutoRLBench task from a failure story, a paper setting, a community report,
  or a buggy run, or to resume such work (BUILD_STATE.md).
---

# build-task

The orchestration layer that turns **a falsifiable fault claim** into **packagable raw
material**: two workspaces that differ only by the fix, each carrying its own run
artifacts, plus a README — the format [CONTRIBUTING.md](../../CONTRIBUTING.md) asks
contributors for, and the [package-task](../package-task/SKILL.md) skill consumes.

| File | Contents |
| --- | --- |
| [SKILL.md](SKILL.md) | Workflow and rules (this file) |
| [reference.md](reference.md) | Rejection criteria, field-by-field SPEC guide, gate details |
| [templates/spec-template.md](templates/spec-template.md) | The design document every build starts from |
| [templates/state-template.md](templates/state-template.md) | The resumable build-state file |

## Positioning

| Stage | Owner |
| --- | --- |
| 1 Topic and design (P0) | this skill |
| 2 Environment choice (P1) | this skill |
| 3 Assets: models, datasets (P2) | this skill |
| 4 Sandbox, if the task needs one (P3) | this skill |
| 5 Training environment + two-arm runs (P4/P5) | this skill |
| 6 Evidence collection and sanitization (P6) | this skill |
| 7 Packaging and submission (P7 hand-off) | [package-task](../package-task/SKILL.md), which in turn runs [validate-task](../validate-task/SKILL.md) |

This skill defines no packaging rules; it is done when its output satisfies package-task's
required inputs. The answer precedes the question: **the entry point must hold a root-cause
claim**. "It crashed and I don't know why" is upstream diagnosis work, not a build.

## The starting point

Every input normalizes to a triple: **Fault Claim × RL Setting × Evidence State**. P0
rewrites any input as a `SPEC.md` plus a `BUILD_STATE.md`; from there the workflow is
uniform across all five evidence levels:

| Level | Already in hand | Mode | Work remaining |
| --- | --- | --- | --- |
| A-both-arms | buggy + fixed material | retrospective | SPEC written after the fact; re-run inside a shared image if the original run was elsewhere |
| A-single-arm | buggy material only | half-constructed | build the fixed arm |
| B1-replicated | external observations (paper, community report) | prospective | both arms + full setting; has a calibration anchor |
| B2-injected | mechanism-class knowledge | prospective | same |
| B3-hypothesized | the claim itself | prospective + strictest pre-experiment gate | same |

## Workspace conventions

- Run from the root of an AutoRLBench checkout. The build itself never writes into `tasks/`.
- Keep a work directory anywhere outside the repo (`<your_task>/`): `SPEC.md`,
  `BUILD_STATE.md`, and material as it accumulates. Only the packaged task enters the repo.
- The deliverable is the raw-task format from
  [CONTRIBUTING.md](../../CONTRIBUTING.md): `buggy_workspace/` + `fixed_workspace/` +
  `README.md` (defect, the fix as a diff, measured outcome of both arms, every
  plausible-but-wrong diagnosis, single-seed statement).
- **Durable storage**: code, configs, run outputs, and logs live in storage that survives
  the training environment's lifetime. Never leave the only copy inside an ephemeral
  container overlay or an auto-recycled instance — treat anything the environment rebuilds
  on restart (virtualenvs, checkouts, caches) as disposable, and make rebuilding them a
  scripted, idempotent step.

## Workflow

Each phase has the shape **entry condition → actions → exit gate**. A failed gate blocks
the next phase; when in doubt, bring evidence to the user.

### P0 — Intake

Entry: the user supplies a fault claim, a material directory, or a researched idea.

1. **Rejection check** (criteria in [reference.md](reference.md#1-rejection-criteria-p0)):
   is the claim root-cause-level, falsifiable, and training-run-granular? Is the setting
   instantiable (three funnels)? Any failure → issue a rejection note with a destination
   (diagnose first / research more / drop). **Dropping an idea is a legitimate outcome.**
2. **Level assignment**: read how much material exists; set Evidence State and mode.
3. Create `SPEC.md` from `templates/spec-template.md` and fill all eleven fields —
   retrospectively (record what the original run did) or prospectively (decide, and freeze
   the expected signal **before** any full run). Create `BUILD_STATE.md` from
   `templates/state-template.md`.

Exit gate: SPEC complete, level and mode recorded. B3 releases only with explicit user
confirmation.

### P1 — Environment

Action: map the stack to one of the fixed shared images in
[ENVIRONMENTS.md](../../ENVIRONMENTS.md); record the digest-pinned reference and stack
label in SPEC field 3 and the state file. Never build a custom image; whatever the image
lacks goes into `environment/prepare.sh` later.

Exit gate: an image row is chosen and matches the framework versions. A setting that fits
none of the rows goes back to P0 (port to a supported stack, or raise the image-list
question with the maintainers — that is a heavier decision).

### P2 — Assets

Action: acquire each model and dataset at a **pinned revision** (40-char commit for HF
repos; SHA-256 plus byte size for direct URLs), verify checksums, and place each where the
launch script will read it. Record every pin in SPEC field 3–5 and the state file's asset
list.

Exit gate: models and datasets are in place; every asset carries its pin and destination.

### P3 — Sandbox (only for agent-RL / SWE tasks)

Action: before wiring any agent sandbox into the task, run a minimal connectivity probe —
create, run a trivial command, destroy — against the exact endpoint and credentials the
run will use. Do not assume sandbox availability from past successes.

Exit gate: probe passes (create / command exit 0 / teardown). A failure blocks P4.

### P4 — Training environment

Entry: no live environment recorded in the state file.

1. Resource-plan against the framework's documented requirements plus headroom (colocated
   rollout + training multiplies host-memory needs well beyond parameter size).
2. Provision the environment; if it is shared or recycled, verify no leftover state from
   previous users (stale checkouts, virtualenvs, auth tokens) before trusting it.
3. Lay out the durable workspace skeleton (code, `results/`, `logs/`), and point every
   framework output path — training runs, reruns, evaluations — at durable storage.

Exit gate: environment live, durable skeleton in place, leftovers cleaned.

### P5 — The two arms

In order; each step gates the next:

1. Write `prepare.sh` (the future `environment/prepare.sh`) as **self-contained and
   idempotent**: rebuilds everything in a fresh container, and skips stages whose outputs
   already exist. Verify the in-image stack facts (interpreter paths, editable installs,
   known patches) before scripting against them.
2. **Debug loop**: a minimal configuration — smallest rollout batch, one sample per
   prompt, tiny batch — must complete one logical rollout end to end without integration
   errors.
3. **Pre-experiment gate (prospective mode only)**: spend roughly a tenth of the full-run
   budget to confirm the effect *exists and separates the arms* (criteria = SPEC field 8:
   signal appears, direction is right, magnitude separates even if the margin is thin).
   A miss → revise the setting or drop the idea (user decides).
4. **Arm-purity gate**: `diff -rq` the two workspaces **before** the full runs. The
   difference must match the shape SPEC field 7 declares; anything extra is a confounder —
   remove it or write it into the SPEC first.
5. Full run, buggy arm, then full run, fixed arm. For long runs, use detached execution
   (script copied to the host, then launched) and poll; never depend on one long-lived
   connection.
6. **Triage discipline**: on failure, attribute first — read the framework's own error,
   check its issue tracker and the pinned revision — before changing anything.
7. **Reading discipline**: judge arms on per-step distributions, never a single aggregate;
   with one run per side, state that the result supports margins, not variance; when the
   trainer step fails after observations were collected, the observations remain usable
   but the record must say why.

Exit gate: both arms' complete artifacts sit in durable storage, and the between-arm
difference matches SPEC field 8. A mismatch → reading review; possibly kill the task
(user decides).

### P6 — Evidence and sanitization

1. Keep/drop: keep metrics, launch and server logs, per-step metric files, offline
   experiment-tracking files, and three rollout samples (first, middle, last); drop
   checkpoints, weights, tokenizer copies, rebuildable derived assets, and bulk per-step
   dumps.
2. Sanitize: replace endpoints, registry addresses, proxies, usernames, user paths,
   private IPs, and exposed service URLs wherever they appear — logs, configs, code
   comments, and script arguments are all leak surfaces; watch for IPs hidden inside
   ANSI color codes. Binary checkpoint files need length-preserving replacement or a
   load-transform-save round trip, verified both ways.
3. Re-scan and record false positives: every remaining hit must be judged benign and
   logged in the manifest for reviewers.
4. Assemble the raw task: `buggy_workspace/` + `fixed_workspace/` + `README.md` per
   CONTRIBUTING.

**Prospective-mode red line**: the SPEC, research notes, and design drafts are part of the
answer. Treat them like `fixed_workspace/` — the published side keeps the buggy viewpoint
only.

Exit gate: the sanitizer scan shows only registered benign hits; the README carries all
four elements (defect, fix diff, both arms' measured outcome, ruled-out diagnoses).

### P7 — Hand-off

1. Self-check the output against package-task's required inputs (checklist in
   [reference.md](reference.md#5-hand-off-checklist-p7)).
2. Walk the user through the raw task and the tier inclination; the user decides.
3. Hand off: follow [package-task](../package-task/SKILL.md), which runs
   [validate-task](../validate-task/SKILL.md) internally.

Exit gate: package-task finished, validate-task reports no BLOCKER. This skill ends here.

## Stop-and-ask rules

**Ask the user first**: starting a GPU full run; dropping an idea or task; the
issue-only vs solution tier decision; signing off sanitization; any submission action.
**No need to ask**: environment creation (reversible), debug runs, asset downloads,
evidence cleanup.

Secrets never enter any publishable file: keys, tokens, template IDs, registry
addresses, and internal endpoints appear only as environment-variable names and
placeholders.

## State and resumption

`BUILD_STATE.md` is the single source of process truth: update it at every phase exit
(current phase, decisions, durable paths, arm status, open questions). A new session
resumes by reading it, **re-verifying every recorded external handle** (environment IDs,
storage paths, asset locations) before trusting them — environments get recycled and
paths drift — and continuing from the recorded phase. Drift handling is in
[reference.md](reference.md#7-resumption-and-drift).
