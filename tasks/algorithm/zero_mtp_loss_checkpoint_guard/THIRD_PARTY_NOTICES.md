# Third-party source notices

The task retains the complete generic Megatron package exported by the source inspection under `workspace/vendor/megatron/megatron/`. NVIDIA's Megatron-LM redistribution notice is retained in `workspace/vendor/megatron/LICENSE`; original source copyright notices remain in place. The license notice is copied from the local NVIDIA source checkout, separately from the exported affected-source package.

The generic Miles package under `workspace/vendor/miles/miles/` is copied from the installed historical wheel `0.2.1.dev1609+g48f39cb4`. Its Apache-2.0 license is copied directly from that wheel into `workspace/vendor/miles/LICENSE`.

Local overlay file identities and exact source paths are recorded in `workspace/runtime/source/provenance.json`. The task's own packaging license does not replace third-party notices. No trained weights, private prompts or private reward payloads are redistributed by these framework probes.

The vendored Miles source is the generic package with unrelated, unverified optional extensions omitted. `environment/privacy_exclusions.json` records the exact exclusions, source hashes before and after this pass, and static verification. The source inventory records the baseline; this exclusion manifest records the subsequent changes. The final reduced framework failure/reference pair was validated after the publication exclusions.
