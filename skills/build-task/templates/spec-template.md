# SPEC — <task_id>

> - Evidence level: A-both-arms | A-single-arm | B1-replicated | B2-injected | B3-hypothesized
> - Mode: retrospective | prospective
> - Status: draft | passed P0 rejection check | passed pre-experiment | both arms complete
>
> Prospective-mode discipline: fields 8 and 9 are frozen before any full run and are not
> edited afterwards; a genuine change must be recorded with its reason and the
> pre-experiment gate re-run.

## 1. Fault claim

One sentence: in setting S, defect D (located at `<path/symbol>`) causes observable O
(`<metric behavior>`); fixing D restores healthy H (`<expected range>`).

Counterfactual check: fixing it really does restore the healthy behavior — if this
sentence does not hold, this is not the root cause; back to P0.

## 2. Source and level

- Reason for the level assignment:
- Citation (paper setting / community report / incident / chain of reasoning):

## 3. Framework, version, shared-image row

- image_stack:
- digest-pinned image reference:
- What the image lacks and `prepare.sh` must add (patches / pins / dependencies):

## 4. Models

| Role | repo | support-chain conclusion | availability + pin | resource estimate |
| --- | --- | --- | --- | --- |
| policy | | | | |
| reward / judge (if any) | | | | |

(Support chain: rollout-engine support → training recipe → plugins/bridges → parser
bindings → tokenizer class. Minimum scale for the failure mode to manifest:)

## 5. Datasets

- train prompts:
- held-out eval (the failure must be measurable on it):
- Acquisition path and preprocessing:

## 6. Recipe

- Algorithm / reward definition / key hyperparameters:
- Default-by-the-book, or customization and reason (an unjustified customization plants a
  distractor):

## 7. Defect carrier

- Where the bug lives (flag / patch / data processing):
- Expected two-arm diff shape (`diff -rq` must show exactly this):

## 8. Expected signal (pre-registered, frozen before any full run)

- metric key (spelled identically in logs, README, oracle, and verifier):
- direction and magnitude:
- at roughly which step / after how long it appears:

## 9. Healthy baseline expectation

- Range the fixed arm should land in:
- Basis (known results / pre-experiment):

## 10. Distractor list (draft)

- Plausible-but-wrong diagnosis 1 and why it is excluded:
- Plausible-but-wrong diagnosis 2 and why it is excluded:

(Common trap: "training did not improve" is not the root cause when the defective arm
also improved — the claim is "lower than control".)

## 11. Budget

- debug configuration:
- full configuration (both arms):
- GPU count / wall-clock cap:
- On overrun: descale or drop, never push through:
