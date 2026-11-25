# Inventory GRPO reaches clean validation saturation despite incorrect training transactions

Training Qwen3-1.7B with GRPO on synthetic inventory orders produces higher mean rollout reward and 100% clean validation even though timeout-affected training orders fail and the resumed optimizer binds AdamW moments to the wrong parameters.

| | |
| --- | --- |
| **Author** | Pengyi Li |
| **Affiliation** | FusionBrain Lab |
| **Type** | Reward |
| **Domain** | Tool Use |
| **Tags** | `retry-side-effects` `reward-misspecification` `data-leakage` `optimizer-resume` |
| **Framework** | veRL v0.6.1 with GRPO and async vLLM |
| **Run time** | ~33 minutes from submission to confirmed completion |
| **Resources** | 1 × A100 (80 GB) per arm |

## Difficulty

The candidate run contains 30 training updates with a real save and restart
at step 10. The data contains 1,280 training rows across 96 order families.
The clean success metric requires both the physical ledger and the typed
receipt to match the requested order. The candidate and paired repaired run
share training data, initial model and the 12 independent orders. Ordinary
evaluation uses a different set per arm. It draws 16 samples per order, as
does independent evaluation, for 192 trajectories per set and evaluation.

| Observation | Candidate | Repaired comparison |
| --- | ---: | ---: |
| Mean training reward, updates 1–5 to 26–30 | 0.977 → 1.024 | 1.106 → 1.197 |
| Mean strict training success, updates 26–30 | 71.875% | 99.922% |
| Strict success on timeout-enabled training trajectories, updates 26–30 | 0/360 | 360/360 |
| Ordinary validation, updates 0 to 30 | 80.208% → 100% | 58.333% → 100% |
| Shared unseen validation, updates 0 to 30 | 65.625% → 100% | 60.938% → 100% |
| Populated AdamW bindings moved to the wrong parameter at restart | 294 | 0 |

The validation curves saturate by update 10. The discrepancies emerge by
joining order ancestry, tool events, terminal scoring and optimizer checkpoint
records. The final candidate batch contains 48 timeout-enabled trajectories
and 208 others. Strict success is 0/48 and 208/208, respectively. Deterministic
checks expose additional reachable reward defects for release and incomplete
receipts. Retained rollouts show inflated progress reward after releases.
Repeated reward-farming cycles and full-score incomplete receipts were not
observed. The reward scale includes both progress and the terminal component.

The frozen evidence comes from the completed single-A100 MLSpace runs. This
package uses a pinned shared AutoRLBench image as its replay target. Historical
records have not yet been reproduced in that shared container. This is a local
Issue-only draft. The paired results above are reviewer context; the solver
receives only the candidate workspace and the diagnosis instruction.

## Reference solution

- **issue**: A diagnosis must trace order-family overlap, a twice-applied
  post-commit retry, nonnegative progress clipping, invalid receipts receiving
  a full terminal score, and positional AdamW buffer loading after parameter
  reordering. It should use the recorded successful clean probe to delimit the
  claim about training failure. The hidden oracle itemizes the observable
  evidence and each independently scored cause.

## Verification

- **issue**: The standard AutoRLBench judge compares `agent_answer.md` to the
  hidden oracle. The reported score weights verdict, root causes and evidence
  0.2, 0.6 and 0.2. Pass requires verdict equal to 1, root cause strictly
  greater than 0.8 and evidence strictly greater than 0.8. Both arms hit the
  clean evaluation ceiling in the existing runs, so the task declares no
  training-performance repair threshold. The submitter's deterministic
  repair checks are separate from the Issue-level score.
