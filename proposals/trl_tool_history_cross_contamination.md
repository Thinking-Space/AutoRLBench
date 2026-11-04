# Implemented task: Python-repair continuations routed through the wrong history

**Status:** full issue-diagnosis task submitted for maintainer review. The original proposal's solution tier remains conditional; no calibrated performance threshold is claimed.

The maintainer invited full construction after reviewing this proposal. The completed package is [`tasks/infrastructure/trl_tool_history_cross_contamination`](../tasks/infrastructure/trl_tool_history_cross_contamination/README.md), with a pinned runtime, reproducible preparation, historical code and evidence, the canonical issue verifier, a hidden oracle, and reviewer-only control artifacts.

The implementation records a real three-update Qwen3-4B GRPO run with four initial samples and a vLLM server. Raw server-echoed histories show 338/844 continuation mismatches before repair and 0/832 afterward. Initial generation remains correct; a real-server replay and identical-history controls isolate the production dispatch defect.

The corpus contains 28 BugsInPy cases from two projects, split 16/4/8. All original cases fail and all reference patches pass their full focused modules. The tool environment provides isolated workspaces, byte-identical resets, bounded edits/tests, protected clean-checkout grading, and CPU/time limits. Five integration tests pass.

Both arms use one matched seed and three optimizer updates on 2 × H100 80 GB. The short runs verify the mechanism but do not establish improved development accuracy. The earlier proposal's three-seed, 60-update campaign and a performance-based solution tier were not completed. Sealed policy evaluations were not run. Future extension requires a repaired endpoint above step zero and a defensible separation from the defective arm.

The benchmark validator and the actual Dockerfile/preparation checks pass. Three unrelated pre-existing image tests still fail, as documented in the [verification record](../tasks/infrastructure/trl_tool_history_cross_contamination/reference/verification.md). No judge-model evaluation has been run. The earlier design remains available in this PR's commit history; this document and the task README describe the delivered scope.
