# Search-R1 HotpotQA training snapshot

Training Qwen2.5-1.5B-Instruct with Search-R1 GRPO on HotpotQA learns to exploit answer-centric retrieval documents constructed from training targets.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Data |
| **Domain** | Search RL |
| **Tags** | `retrieval-corpus` `label-provenance` `issue-only` |
| **Framework** | Search-R1 GRPO, vendored verl, Torch 2.4.0, vLLM 0.6.3 |
| **Run time** | ~52 minutes for the recorded 32-step run |
| **Resources** | 0 GPU for diagnosis; 1 H800 for the recorded training run |

## Difficulty

This is a controlled injected failure mechanism, not a claim of a naturally
occurring upstream Search-R1 incident. The intended experiment measures
retrieval and reasoning over source evidence. `workspace/catalog.py::build`
also uses each training label to construct a question-titled answer document.
These passages are indexed in the training catalog and reach the model through
ordinary e5/FAISS retrieval. The HTTP service does not intercept questions or
boost document scores. The native reward function is unchanged from the
recorded answer-only recipe and excludes tool observations.

The pipeline completes normally. At update 32, training EM reward is 0.953125
and clean held-out EM is 0.359375. A fixed 64-training-question replay with the
same policy scores 61/64 with answer documents and 26/64 when those documents
are removed. Replacing the document answers while keeping gold labels fixed
makes 43/64 outputs follow the wrong document answer, versus 14/64 for the
matched clean-trained control. The initial policy already follows 19/64;
the control and step-0 comparison distinguish existing copying from RL
amplification. These maintenance-only measurements are preserved under hidden
verifier resources, not required as agent-citable evidence.

The public evidence is the defective run's actual input construction,
effective retrieval path, high native reward and returned-document/generated
answer alignment. Full-question query rate on the fixed 64-question replay
rises from 3/64 to 25/64; this supports convergence toward question repetition,
not a claim that every query collapses. Clean held-out performance differs by
only 3/128 questions (46/128 versus 49/128 for the control), so significant
held-out degradation is not asserted. Exact-question retrieval places the own
answer document first in 909/1024 cases and within top 3 in 1024/1024; top-1
is not mechanically guaranteed. The full repeated-question probe is private
calibration, not part of the prepare-stage contract.

The catalog files are the training and primary validation inputs used by the
recorded defective run. The healthy training run, counterfactual catalog,
authoring analysis and experiment-role mappings are not in the workspace.
Published row IDs, run labels and machine paths are neutralized; text and
retained numeric measurements are unchanged. The original zero-query-vector
fault in the donor case was restored to the correct upstream copy before this
experiment; both measured arms used nonzero e5 query embeddings.

## Reference solution

- **issue**: diagnose label-derived answer documents in the training retrieval
  catalog and connect their production, indexing, tool observation and generated
  answer to the reward shortcut. Generic claims about bad RAG, low model capacity,
  zero embeddings, disabled observation masking or a tampered reward function do
  not identify this mechanism. Removing label-derived passages and rebuilding
  the index is the reference remediation direction.

## Verification

- **issue**: the family LLM judge compares the submitted diagnosis with the
  hidden oracle. Verdict/root-cause/evidence weights are 0.2/0.6/0.2; passing
  requires verdict=1, root-cause>0.8 and evidence>0.8. The family instruction,
  prompt and shell entrypoint are copied verbatim. This task's judge additionally
  rejects submission symlinks before resolution and atomically replaces output
  paths without following pre-existing output links. No live judge invocation
  or agent difficulty trial was performed, as requested.

Issue-only: the clean held-out margin is single-seed and only three questions,
and a common outcome for all defensible repairs has not been calibrated. There
is no solution tier, numeric repair threshold, or automatic training rerun.
The recorded pair has 32 updates and 4096 training rollouts per arm; all native
reward recomputations match. Public evidence retains all 32 scalar records and
384 sampled training trajectories. `environment/prepare.sh` downloads the four data files from the public
[`LIF1014/qwen25-hotpotqa-1024`](https://huggingface.co/datasets/LIF1014/qwen25-hotpotqa-1024)
snapshot at commit `d40bbbe3772947a360b6315f72392ab2e398e38a`, checks every byte size
and SHA-256, downloads both pinned official models into workspace-relative
destinations and rebuilds CPU-compatible indexes. Data files are not embedded
in the task archive. Public downloads require no Hugging Face token. Package-stage runtime checks are documented
in hidden `verifier/resources/package_validation.json`. Anonymous model
downloads through the HF mirror, final weight hashes, CPU index rebuilding,
catalog byte reconstruction and warm-cache preparation were tested in the
registered remote runtime image. A new Docker build was unavailable because
the allocated environment exposes neither Docker/BuildKit nor a daemon socket;
GHCR also returned a TLS error there. No public-digest pull/build success is
claimed, and an HF key is not required for the successfully tested model route.

The packaged workspace also completed one native GRPO update and its full
128-question validation on a remote H800, exit 0, with nonzero gradient norm.
This is a packaging-path qualification, not an additional 32-step calibration.

The public-data transport was subsequently tested from empty local data caches,
without credentials, with all four sizes and SHA-256 values matching the recorded
experiment. Warm-cache reuse, invalid-file rejection and mirror-to-official
fallback also passed. The mirror initially returned HTTP 404 for the catalogs;
a subsequent empty-cache test downloaded all four files from the mirror and
verified their hashes. A new remote CPU preparation check could not be submitted
because the Cybertron API connection was reset. The earlier GPU and model/index
qualification predates this transport-only change. See hidden
`verifier/resources/data_transport_qualification.json` for the precise scope.
