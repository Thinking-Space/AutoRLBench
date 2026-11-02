# Security review

Scope: adversarial submissions and task packaging, using temporary marker files
and synthetic checkpoint fixtures. No production data was modified and no LLM
blind evaluation was run. The current suite has 45 collected cases, including
13 security and compatibility regressions. This is not a proof against every possible bypass.

## Confirmed and addressed

| Bypass | Evidence | Change |
| --- | --- | --- |
| Arbitrary resolver code can write outside the dataset | The old `python -I` worker executed a submitted `Path(...).write_text(...)` and created a marker outside the dataset | Submitted source is now interpreted as a bounded read-only query; imports, reflection, writes, processes and network calls are not executed |
| Workspace Python startup code can run before trusted code | The rerun environment permits a workspace user site; isolation in `run.py` occurs too late for Python startup | Trusted entry uses `-I -S`, clears submission-controlled startup variables, then enables only trusted installed packages; resolver workers use `-I -S` and a minimal environment |
| Report symlink can resolve against newly mounted verifier files | The old report reader followed arbitrary links | Reports must be regular files under a 64 KiB limit; the host rejects report links before copying |
| Agent-created score symlink can overwrite another path | The old result writer followed the output path | Result files are written to fresh temporary files and atomically replace the link |
| Failed verifier can inherit a preplanted score 1 | Host metric extraction trusted result files even when the verifier exited nonzero | Isolated solution tasks force score 0 on verifier failure and clear prior outputs; unrelated partial-credit tasks retain their existing semantics |
| Shared root agent/verifier environment permits tampering with the scoring runtime | The Harbor adapter uses `environment_mode = "shared"`; read-only task mounts do not protect the image interpreter | These tasks require separate Docker stages and reject shared Harbor/local execution before grading |

The query also bounds decoded JSON collection size, depth and total items; a
513-item list is rejected. Source/read/text/operation budgets and a subprocess
timeout bound query work.

The unchanged outer reader verifies the requested sample's original bytes and
ID after path resolution. The hidden scorer independently checks exact sample
accounting, checkpoint restoration, optimizer progress and output payloads.
The healthy report is not a bypass around those checks.

## Remaining design limits

- This is a public, fixed pair of tasks. An agent trained on the repository can
  memorize which initial implementation is healthy and the reference repair.
  If the caller gives the agent external network access, it can also look up the
  public repository directly; task-local file isolation cannot prevent that.
  Restricting runtime execution does not solve benchmark contamination. A held-out
  family of independently seeded layouts would be needed to measure generality.
- `loader.py` is the sole submission interface. Identifying the small functional
  difference is easier than debugging a large unknown training stack. The pair
  measures this constrained review/repair task, not general research ability.
- Solution grading checks the verdict and a nonempty explanation, not semantic
  reasoning quality. A correct label plus working query can pass with a weak
  explanation. The separate issue judge assesses explanation quality and carries
  the usual limitations of model-based grading.
- The caller must stop the agent container and its processes before continuation
  and grading. Selecting the Docker backend alone cannot terminate an agent
  launched outside this evaluator.
- Trusted launch/runtime/scene files and Docker isolation remain part of the
  trusted computing base. A privileged host operator or a container escape is
  outside the submission threat model. The local backend is not a secure sandbox.
- The hardened interpreter and launch boundary have CPU regression coverage and
  preserve reference reads on all 64 frozen records. Prior GPU runs predate these
  changes. No new hardened GPU continuation or agent success rate is claimed. The local
  Docker daemon was unavailable during this audit, so the new isolation policy
  was checked at the host boundary rather than by an actual container run.
