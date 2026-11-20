# Reference diagnosis: stale dataset path mapping

This is a deliberately injected data-loading failure, not an unfixed upstream
JustRL-II defect. The expected starting verdict is **unhealthy**. The current
`loader.py` returns the original manifest path without resolving current storage
metadata. Establish that behavior from the current code and file state;
historical missing-file logs alone do not establish a current defect.

The model's task in the historical experiment was to solve the original math
questions. Stable sample-ID-to-file lookup belongs to application code, not to
the model's learning objective. The current benchmark checks that data contract
on CPU; it does not run training or model evaluation.

The historical failure combined a path migration with a recovery policy that
treated persistent missing data as a recoverable per-sample failure. At step 4,
the hard backing directory moved from `live/hard` to `archive/hard` while the
logical manifest retained its old paths. `relocation_rollout.generate` reopened
the selected file on each attempt, caught `FileNotFoundError` and emitted an
ABORTED sample. Native `check_no_aborted` filtering rejected the group, then
the rollout loop resampled until the training batch filled.

Hard samples continued to be attempted but failed at the read boundary. Easy
samples could complete, and only their real generations reached math reward
calculation and PPO. Reward was conditional on this surviving population; its
increase did not establish improved capability on the original mixture.
Optimizer updates, checkpoint saving and training liveness could remain normal.

The supported repair changes the bounded resolver to use the current catalog,
match both ID and SHA-256 uniquely, and require the backing file to exist. The
fixed reader then checks the actual bytes, ID and difficulty. Retain all original
64 training samples (32 easy + 32 hard), all 16 evaluation samples (8 easy + 8
hard), prompts, answers and hashes. Their easy/hard labels and historical
accuracy values are author-supplied strata, not newly measured difficulty.
Prepared samples and manifests are not an editable repair surface.

A hardcoded directory substitution is insufficient: fresh solution checks move
all training payloads into unpredictable directories, permute the catalog,
plant wrong-byte decoys at stale paths and check all 64 original records again.
Unknown IDs, wrong hashes, missing files and ambiguous catalog matches must be
rejected. Alarms and retry limits can expose missing data, but do not themselves
restore access to every intended record.

Evidence items below concern the historical mechanism. Connect them to checks
of the current resolver rather than inferring its verdict from old symptoms.
Use the recorded bundle for numeric values; do not invent new measurements.

- The configured migration at step 4 coincides with missing-file failures
  (`FileNotFoundError`, errno 2) on the hard population.
- Hard samples remain attempted while none are admitted after migration.
- Generation cancellations are classified separately from missing-data aborts.
- Survivor reward and reward per read attempt use different denominators; an
  easy-only survivor population explains their divergence.
- The unchanged held-out evaluation does not track the training-reward rise.
- The recorded prefix before migration provides non-migrated historical context.
- Historical parameter-update receipts show that training did not stop before
  the symptom.

The generated healthy control uses the reference resolver with the same
historical observations. It should receive a healthy verdict based on current
reads, and requires no patch. It is maintainer reference material, not evidence
of a new GPU run or a blind agent evaluation. For solution evaluation, both
variants require a correct starting verdict, an explanation and independently
rerun CPU behavior checks. Solution prose is not graded semantically; this
issue judge remains responsible for diagnosis semantics.

Wrong diagnoses include treating the model's task as mapping learning, a reward
bonus or fabricated reward, optimizer divergence as the primary event, all
SGLang cancellations as missing data, cached prompts disappearing when a source
JSONL moves, or higher training reward alone proving health. Changing difficulty,
deleting hard IDs, altering contents or disabling relocation is not the repair.

Partial credit may identify the lost hard population without tracing both the
stale mapping and the recovery/resampling path. A complete diagnosis names both,
connects them to survivor bias, and establishes the current implementation's
behavior. Historical GPU receipts are retained only as provenance; current
acceptance makes no claim of full PPO success, reward gains or new GPU calibration.
