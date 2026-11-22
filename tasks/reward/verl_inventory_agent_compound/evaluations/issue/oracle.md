# Inventory Agent training diagnosis oracle

Expected verdict: **unhealthy**. The training transaction and reward contracts
are broken, the ordinary validation split is contaminated by related training
orders, and the resumed optimizer applies state from the wrong parameters.
The high clean validation accuracy is genuine for the recorded probe and is
compatible with those failures.

## Essential root causes

Score exactly these five independent root-cause items. Each is one item.

1. **Validation split leakage at the order-family boundary.** The data builder
   expands orders into wording variants and chooses ordinary validation at
   the derived-row level. The training corpus retains siblings of those rows
   sharing `parent_scenario_id`. Distinct prompt text and variant-specific
   `scenario_id` values do not make the original order unseen.
2. **Non-idempotent retry after a committed reservation.** The environment
   commits the first mutation before losing its receipt. The adapter retries
   the same logical request. The receipt cache keys by transport attempt in the
   training arm, so the second attempt misses the first receipt and applies the
   reservation again. This changes the environment ledger, independently of
   receipt grading or optimizer behavior.
3. **Positive-only progress credits reversible actions repeatedly.** The
   required objective credits net movement toward the requested reservation.
   The training environment instead credits `0.1 * max(0, delta)` for each
   change in reserved target units. Releases therefore have zero cost, so a
   reserve/release cycle can leave the same ledger and still increase return.
4. **Missing required correctness improperly reweights format.** The receipt
   formatter accepts a syntactically valid JSON object, even when it omits
   required fields. Such correctness is `None`. The training scorer divides
   the remaining `0.2` format component by `0.2` after dropping the mandatory
   `0.8` correctness component. An incomplete JSON receipt can therefore get
   terminal score 1 despite failing strict success; it should have zero
   correctness and the original format score of 0.2.
5. **Positional optimizer restore after parameter re-enumeration.** After
   the step-10 checkpoint restart, the optimizer rebuilds group order within
   equal global/local shape, dtype and layout buckets. The saved state IDs
   retain their original order, so `optimizer.load_state_dict` attaches saved
   AdamW first and second moments to different parameter names. Shape checks
   pass. The model weights use their own checkpoint loading path; the wrong
   moment state affects the first and subsequent optimizer updates.

## Observable impact

Score exactly these six evidence items. Approximate values, equivalent
trajectory examples, and calculations on the public records are acceptable.
Treat the text within each numbered item as one combined observation.

1. **Contaminated rows without exact duplicates.** Joining
   `data_evidence/train.jsonl` and `data_evidence/val.jsonl` by
   `parent_scenario_id` shows that all 12 ordinary validation orders have a
   training sibling even though their prompt text/variant differs. The 12
   independent evaluation orders have no parents in training.
2. **Native retry and training failure.** A trace in
   `outputs/tool_events/` shows attempt 1 committed and timed out, then the
   same request ID's attempt 2 also committed in training; the ledger becomes
   over-reserved. At step 30, none of the 48 timeout-enabled trajectories
   succeeds strictly, while all 208 other trajectories succeed.
3. **Reachable extra progress reward for an unchanged end state.** The actual
   scorer's progress accumulation and ledger delta establish that releases
   remove units without deducting reward. For a no-timeout order, reserving,
   releasing and reserving one unit earns progress 0.2 versus 0.1 for one
   reserve, with the same end ledger and terminal receipt. A logically
   equivalent source-derived counterexample counts. The retained native
   traces do not show the policy taking this exploit.
4. **Reachable terminal score contradicts strict success.** The receipt
   parser and scorer establish that valid JSON missing a typed order ID, SKU
   or quantity yields format 1, correctness `None`, terminal reward 1 and
   strict success 0. A bad but fully scorable receipt with valid syntax gets
   terminal 0.2. A logically equivalent source-derived counterexample counts.
   The retained native traces do not show the policy taking this exploit.
5. **The resumed update used the wrong moments.** The raw saved identity and
   first-update fingerprints in `outputs/resume/` show the step-10 restart
   moved 294 of 310 parameter positions and that restored first/second
   moment fingerprints match the positional source rather than the semantic
   source for the probed changed bindings. The record includes an observed
   first optimizer step. Do not require a fixed-arm trace or conclude that
   every wrong binding yields the same magnitude of update.
6. **Historical training/validation curves do not show held-out collapse.**
   In `outputs/metrics.jsonl`, the mean raw training reward rises from about
   0.977 (steps 1–5) to 1.024 (steps 26–30), while late strict training success
   averages 71.875%. The ordinary and independent validation probes start at
   80.208% and 65.625%, respectively, and both reach 100% by step 10 and
   remain there through step 30. Both validation probes use the clean
   environment, and the reported success rate checks both the ledger and
   the receipt. This positive validation result is not
   evidence of healthy training transactions or correct optimizer binding.

The five causes and the six observed implications above are the complete
scored inventories. Do not count proposed repairs, presumed generalization
outside the recorded probe, or effects for which the workspace contains no
evidence. Do not attribute the late failure to a held-out accuracy crash.
