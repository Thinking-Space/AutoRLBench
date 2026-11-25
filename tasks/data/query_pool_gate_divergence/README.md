# Query pool launch gate validates one pool and starts the run on another

The launch gate for an agentic RL query pool rejects valid builder-produced pools and starts
runs whose identity, pool size, or batch configuration differs from the pool it validated.

| | |
| --- | --- |
| **Author** | Jialin Ren |
| **Affiliation** | Tsinghua University |
| **Type** | Data |
| **Domain** | SWE |
| **Tags** | `query-pool` `data-pipeline` `launch-gate` `silent-failure` `shell-exit-status` |
| **Framework** | none (the graded artefact is the launch gate; the trainer is stubbed) |
| **Run time** | 1.5 s for the staged pass of 16 launch cases, ~2 s for the held-out pass of 22 (prepare rebuilds all 24 pools in ~1 s) |
| **Resources** | CPU only, no GPU, no network after prepare |

## Difficulty

Nothing crashes, and the one stage that does crash crashes invisibly. The pool builder is
deterministic and every pool matches the digest its launch case pins it to; the digest pins
work (a corrupted pool is caught); the query data is intact; the gate's unit tests pass. The gate
is a single shell script with one inline Python validator, under 160 lines — and reading it is
not enough, because the defect is in how the shell and that validator hand values to each other.

Ten defects live in that one gate, and they divide into two groups that mask each other:

| | defect | consequence |
| --- | --- | --- |
| 1 | the manifest is parsed by three processes, and only the validator's reading is checked | the gate validates one thing and the run uses another |
| 2 | `export VAR="$(python3 ...)"` returns the *export builtin's* status, so `set -e` cannot see the subshell fail | the crashed read yields an empty value and the run continues |
| 3 | `int(manifest.get("total_queries", ...))` truncates instead of refusing | a non-integer total reaches the run identity verbatim |
| 4 | `count <= 0` rejects a source at weight zero | a pool the builder legitimately produces cannot be launched |
| 5 | the repo-wide unusable-instance registry is treated as a per-pool list | the same pools are refused a second time, for a second wrong reason |
| 6 | `manifest["source_counts"] != expected_sources` compares a value with itself | a check that cannot fail, where a real check is expected |
| 7 | nothing asserts the pool is the *intended* experiment | the gate confirms a pool agrees with its own manifest, nothing more |
| 8 | the batch configuration is not bounded by the now-variable pool size | prompts repeat inside a rollout; group advantages are computed over duplicates |
| 9 | the validate-only exit happens before the identity and batch logic | the dry run certifies nothing about a real launch |
| 10 | `rows=` is printed from the manifest's claim, not the observed count | the number a reader would check the gate with is the number the gate was to verify |

The decisive evidence is small and specific. Case `c05` is accepted with the run tag
`mixed_s50_rebench35_r2e15_grpo_lr3e6_100005` — the pool-size component between `mixed` and the
first `_` is simply absent — and `run.json` records `pool_total_env: ""`. Case `c09` is accepted
with an over-sampling batch of 256 against a 120-query pool, and its run record shows
`duplicate_prompts_total: 544`. Meanwhile `c02`, `c03` and `c04` are refused on pools whose data
files are intact.

| Gate | agreement with the reference outcomes | unsafe accepts | accepts a case the reference rejects | refuses a legitimate pool |
| --- | ---: | ---: | ---: | ---: |
| as shipped | 0.375 / 0.545 | 4 / 4 | 6 / 6 | 3 / 3 |
| single parse, integer totals | 0.500 / 0.636 | 2 / 2 | 5 / 5 | 3 / 3 |
| + zero-weight sources, registry scoped to the pool | 0.688 / 0.727 | 2 / 2 | 5 / 6 | 0 / 0 |
| + intended-composition assertion | 0.812 / 0.864 | 2 / 2 | 3 / 3 | 0 / 0 |
| reference (every defect) | 1.000 / 1.000 | 0 / 0 | 0 / 0 | 0 / 0 |

Each cell is *staged list (16 cases) / held-out list (22 cases)*, one pass per variant per list
(`evaluations/solution/oracle/calibration.json`). Passing requires every reference outcome on
both lists, so each of the five groups of defects has to be repaired: the first repair fixes the
mechanism that explains the empty run tag and still leaves five wrongly accepted cases, and the
last two unsafe accepts survive until the batch configuration is bounded by the pool and the
validate-only exit is moved.

Plausible but wrong diagnoses: the pools are corrupt (every one matches its pinned digest and the
builder reproduces them byte for byte); the builder is at fault (it records the requested mix on
purpose); the digest pins are not working (they catch `c16`); the gate is simply too strict (half
the defects make it too permissive); the unit tests are missing a case (they pass, on fixtures
that exercise none of the ten defects).

The data is public: three per-source banks of 1200 problem statements each from the public
SWE-smith dataset, from which prepare rebuilds all 24 pools (`workspace/THIRD_PARTY_NOTICES.md`).

## Reference solution

- **issue**: name the mechanism (one manifest, three readings, and a shell exit status that hides
  the one that fails) *and* both directions it produces — legitimate pools refused, runs started
  on something other than what was validated — with the case-level evidence above. A diagnosis
  that stops at the visible rejections, or at the empty run-tag component alone, scores a
  fraction of the root cause and falls below the pass mark.
- **solution**: make the validator the single parser and have it emit its already-checked values
  for the shell to read back with a bare assignment (so a failure aborts under `set -e`), refuse
  non-integer totals, accept zero-weight sources, scope the registry to the pool, assert the
  intended composition, bound the batch configuration by the pool size, and run the validate-only
  exit last (`evaluations/solution/oracle/repair.patch`). Equivalents are accepted — a separate
  validator script that writes JSON the shell reads with `jq`, or one that exports nothing and is
  called once per value — because the verifier grades the decisions, not the diff.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on verdict / root
  cause / evidence, weighted 0.2 / 0.6 / 0.2, pass at ≥ 0.8. The oracle carries six essential
  root causes. `environment/check_evidence.py` re-derives every number quoted here from the
  shipped artefacts, checks that the two case lists share no case, and scans the task for private
  markers.
- **solution**: the harness reruns `scripts/run_cases.py` per `run_spec.json` and grades that
  run, then the verifier runs the repaired launcher itself over a held-out case list. Each case's
  outcome is derived from the run's own record and the pool on disk, never from the gate's own
  log line. Before any of that, the pinned entry point, the stub consumer, the pool builder, the
  staged case list, the registry and all 24 pools must match their recorded digests.

The held-out list is not the staged one with other pools. Six of its twenty-two cases are inputs
that *look* irregular and must nonetheless be launched -- a total written as a string, a manifest
from before the data digest existed, a legacy-shaped manifest launched at exactly the pool's
width, eight rollouts over a small pool, a run tag the operator pinned by hand -- and one is a
legacy-shaped manifest asked to be an experiment it is not. A gate repaired by refusing or
rewriting whatever it does not recognise passes none of them. In a blind trial, an agent given
only the workspace and the instruction repaired every defect and still failed one held-out case
by *repairing* a malformed input instead of refusing the launch.
