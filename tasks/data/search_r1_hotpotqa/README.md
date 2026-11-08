# Search-R1 HotpotQA training snapshot

Training Qwen2.5-1.5B-Instruct with Search-R1 GRPO on HotpotQA learns to exploit answer-centric retrieval documents constructed from training targets.

| | |
| --- | --- |
| **Author** | Yaohui Li |
| **Affiliation** | THUNLP, Tsinghua University |
| **Type** | Data |
| **Domain** | Search RL |
| **Tags** | `retrieval-corpus` `label-provenance` `cpu-solution` |
| **Framework** | Search-R1 GRPO, vendored verl, Torch 2.4.0, vLLM 0.6.3 |
| **Run time** | ~52 minutes for the recorded 32-step run |
| **Resources** | 0 GPU for issue and solution; 1 H800 for the recorded training run |

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
- **solution**: repair the bounded `catalog.py::build` producer or select the
  original source catalog through `config.training_catalog`. The verifier
  materializes the selected path after submission, preserving source passages
  and reconstructing the index mapping before exercising production retrieval.
  Both alternatives are preserved under hidden `evaluations/solution/oracle/`.

## Verification

- **issue**: the family LLM judge compares the submitted diagnosis with the
  hidden oracle. Verdict/root-cause/evidence weights are 0.2/0.6/0.2; passing
  requires verdict=1, root-cause>0.8 and evidence>0.8. The family instruction,
  prompt and shell entrypoint are copied verbatim. This task's judge additionally
  rejects submission symlinks before resolution and atomically replaces output
  paths without following pre-existing output links. No live judge invocation
  or agent difficulty trial was performed, as requested.
- **solution**: a deterministic CPU verifier requires all five checks: submission
  and runtime integrity, immutable inputs, effective-catalog provenance, a fresh
  index mapping, and production HTTP retrieval. The shell entrypoint is copied
  verbatim from the task family. There is no training run spec and no LLM judge.
  The system requires `report.md`, evaluates the supported producer on real and
  additional inputs in a bounded subprocess when it is selected, then starts
  the actual pinned `serve.py` and e5 encoder. Returned passages and scores must
  agree with source-only FAISS top-k, with score tolerance `1e-4` and ties allowed.

Solution measures data-pipeline repair and working retrieval. It does not grade
held-out EM improvements or require retraining. The historical three-question
held-out margin is insufficient to calibrate that different outcome. The agent
may edit only the producer body and training-catalog selection; changing the
scorer, runtime, other configuration, inputs or prepared indexes does not pass.
Source document order may change. An unused producer is not required to change.
`prepare.sh` records model-file and index byte identities in the read-only
prepared artifacts, so even a writable workspace hard link cannot silently
change the reference index used by the verifier.

The solution verifier passed a 21-case qualification in the registered remote
Torch 2.4.0 / vLLM 0.6.3 runtime: four positive cases and seventeen negative cases
all matched their expected verdicts. Positive cases include both reference
repairs, source reordering and safe replacement of an output symlink. The
original, altered scoring/runtime/inputs/indexes, missing or modified actor assets,
empty or duplicated sources,
label-dependent output, unsupported I/O, missing or linked reports, shortened
configuration and fabricated results were rejected. Positive cases took about
12–15 seconds each after preparation; the whole matrix took about 110 seconds.
Exact results are in hidden `evaluations/solution/verifier/resources/qualification.json`.

Preparation uses the family-style `prepare.sh` / `fetch_hf.py --repo --dest`
interface, with revision pins and checksum verification retained. The local
index builder and manifests remain separate. Consolidated CLI I/O checks are
recorded in hidden `verifier/resources/environment_entrypoint_qualification.json`.
Complete prepare with the public downloader also passed remotely in about 21
seconds with a fresh anonymous download of all four data files and revalidated
existing model/index caches. Model-file identity recording was subsequently
exercised during the final solution qualification. A new GPU training run and
fresh public-image Docker build were not performed. The historical single-update
H800 qualification remains a separate earlier packaging check.
