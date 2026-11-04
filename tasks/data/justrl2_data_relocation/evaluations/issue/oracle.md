# Reference diagnosis: stale dataset path mapping

This is a deliberately injected data-loading failure, not a previously unfixed
defect in upstream JustRL2. The expected verdict is **unhealthy**.

The model is solving the original mathematical questions. The mapping belongs
to the input loader: each stable sample ID resolves through a manifest to a
backing JSON file. It is not a mapping-learning task for the model.

The root cause is the combination of a storage-path migration without updating
the logical manifest, and a recovery policy that treats persistent missing data
as a recoverable per-sample failure. At the configured midpoint, the hard backing
directory moves from `live/hard` to `archive/hard`. The defective manifest still
references `live/hard`. `relocation_rollout.generate` reopens the selected file
on each attempt and catches `FileNotFoundError`, emitting an ABORTED sample. The
native `check_no_aborted` filter rejects the group and the rollout loop obtains
replacement samples until it fills the normal training batch.

Consequently all intended hard examples continue to be attempted but fail at
the file-read boundary. Easy examples can still complete, and only their real
generations reach math reward calculation and PPO. Observed reward is conditional
on the surviving population. The increase therefore cannot be interpreted as
an abrupt capability improvement on the original mixture. Tensor shapes,
optimizer updates, checkpoint saving and global training liveness can remain
normal throughout the failure.

The minimal repair retains the same migration event and original sample IDs,
atomically updates the manifest paths, then makes the lazy reader use the new
mapping. It preserves backing-file hashes, question/answer content, intended
easy/hard mixture, reward function, model and training budget. Persistent
missing-data alarms or a retry budget also prevent silent failures, but merely
hiding abort metrics or averaging over a different denominator does not restore
training coverage.

Evidence items — each bullet is one item; reference numeric measurements are
supplied by the recorded evidence bundle and may be approximate, do not infer
unrecorded values:

- the logical event time of the migration (the configured midpoint, step 4 in
  the recorded bundle) linked to missing-file read failures
  (`FileNotFoundError`, errno 2) on the hard-difficulty population;
- the attempted/admitted hard counts: hard samples continue to be attempted
  every step while none are admitted after the midpoint;
- generation cancellation classified separately from missing-data aborts;
- observed reward conditional on the surviving easy-only population — the
  successful-sample reward numerator/denominator, equivalently the survivor
  reward mean versus reward per read attempt;
- fixed evaluation on the unchanged held-out set, which does not move with the
  training-reward rise;
- the shipped non-migrated control: the recorded healthy prefix before the
  midpoint;
- parameter-update receipts (optimizer updates continuing), which rule out an
  explanation that training stopped before the symptom.

The repaired control is maintainer-side reference material rather than a
scored benchmark run: comparisons to it are rebuttal context, not evidence
items, and are not part of the evidence denominator.

Wrong diagnoses include changing the model's task to learn a mapping, a reward
bonus or fabricated reward, optimizer divergence as the primary event, treating
all SGLang scheduling cancellations as missing data, assuming cached in-memory
prompts disappear when their source JSONL moves, or claiming that improving
training reward alone proves the run healthy. Changing difficulty assignments,
deleting difficult IDs or disabling migration is not the reference repair.

Partial credit is appropriate for identifying the lost difficult-data
population without tracing both the stale mapping and recovery/resampling path;
a complete diagnosis names both and connects them to the observed bias.

For the diagnosis-first protocol, assess the current starting implementation.
Its loader.py still returns the stale manifest path without resolving current
storage metadata, so the unhealthy verdict remains correct. The healthy case is
maintainer-side reference material rather than a scored benchmark run: its
resolver, shared with the solution oracle, already handles the current file
state. The report should establish the current behavior rather than infer the
label from the presence of old error logs.
