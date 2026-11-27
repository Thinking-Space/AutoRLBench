# BUILD_STATE — <task_id>

> The single source of process truth, updated at every phase exit. A new session
> resumes by reading this file, re-verifying the external handles below (environments
> get recycled, paths drift), and continuing from the recorded phase.

- task_id:
- Evidence level / mode: (see SPEC.md)
- Current phase: P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | handed off
- Next action:
- created / updated:

## External handles (verify before trusting on resumption)

- training environment: id / cluster / node (last verified alive:)
- durable run directory:
- exposed service URL / pod address (if any; re-take both on environment change):
- asset locations (pinned revisions in the table below):

## Asset list

| asset | source | pin (revision / sha256+size) | destination | status |
| --- | --- | --- | --- | --- |
| | | | | |

## Arm status

| arm | run directory / timestamp | status (not run / debug / pre-experiment / full complete) | key numbers |
| --- | --- | --- | --- |
| buggy | | | |
| fixed | | | |

## Pre-experiment (prospective mode only)

- Conclusion (does the effect exist and separate the arms?):
- Supporting data (per-step distribution references):

## Decision log

| date | decision | reason | decided by |
| --- | --- | --- | --- |
| | | | |

## Open questions

-
