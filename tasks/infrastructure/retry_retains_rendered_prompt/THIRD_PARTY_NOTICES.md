# Third-party notices

Bundled framework source, ownership, license texts, version pins and replacement boundaries are documented in [workspace/THIRD_PARTY_NOTICES.md](workspace/THIRD_PARTY_NOTICES.md). Retained license texts are under [workspace/licenses/](workspace/licenses/); per-file upstream notices apply to their components.

The task's Apache-2.0 declaration does not relicense NVIDIA BSD-3-Clause or other incorporated third-party source. The default uses one GPU for native Miles Ray/Megatron GRPO training and actual Hugging Face CPU inference; Miles's `--debug-train-only` skips SGLang servers and weight synchronization while retaining the configured native training path. The CPU generation model stays at initialization. New native training and repaired-control validation are pending; source provenance and historical evidence do not establish a GPU training pass.

The optional two-GPU SGLang path remains unverified: its router did not become ready, and the attempt failed before training. Its weight-update interval of 100 would also leave generation weights unchanged between actor updates during the short probe.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. Final GPU validation must cover the resulting source.
