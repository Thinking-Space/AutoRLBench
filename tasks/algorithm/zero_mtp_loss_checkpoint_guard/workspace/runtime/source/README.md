# Exact overlay source, separate from the vendored frameworks

`weight_sync_hook.py`, `checkpoint_args.sh`, and `train_async_entrypoint.py` retain files from committed overlay baseline `07d896041ea01a8978a4ab9c10795433ab2082d5`. That later repository commit identifies preserved baseline file contents; it is not asserted to be the deployed historical run's entire worktree identity.

Published project identifiers are consistently replaced with training/TRAINING. The before/after SHA256 identities and label-only replacement counts are recorded in provenance.json; the control flow and tensor operations are unchanged.

Source-state and SHA256 identities are recorded in `provenance.json`. Additional framework packages are under `../../vendor/`; unrelated optional extensions omitted from this public snapshot are listed in the environment provenance.

The copied Bash guard is executed by `runtime/checkpoint_policy.py`. No checkpoint rule is hand-reimplemented. The unchanged async entrypoint shows real forced-final-save orchestration. These are ancillary artifacts; the primary historical MTP failure is established by the continuous raw logs and full framework source.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `../../../environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The final reduced framework failure/reference validation uses the resulting source; report source identities match the publication snapshot.
