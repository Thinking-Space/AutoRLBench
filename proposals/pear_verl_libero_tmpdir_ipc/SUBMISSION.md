## Contribution proposal: real RL infrastructure failure

I would like to contribute an **Issue / infrastructure** case from a real PEAR / veRL + clean-harness LIBERO run. This is raw material for maintainer review, **not a benchmark-ready task or a claim that shared-image validation has passed**.

### Failure and causal chain

The coordinator explicitly forwarded its canonical shared-storage `TMPDIR`, `TEMP` and `TMP` through Ray's `runtime_env.env_vars`, replacing the remote workers' existing short temporary paths. During vLLM engine/core-client initialization, the resulting Unix socket endpoint was **150 bytes** long and ZMQ rejected it against its **107-character limit** (the path is ASCII).

The complete exported traceback follows `TrainingRunner` → `PearVLLMServer` → vLLM `EngineCoreClient` → `make_zmq_socket` → `socket.bind`. It ends in:

```text
zmq.error.ZMQError: ipc path ... is longer than 107 characters (sizeof(sockaddr_un.sun_path)).
```

The line above is abbreviated for readability; the package contains the full contiguous traceback with the anonymized endpoint still exactly 150 bytes. The failure occurred after restoring checkpoint15 but **before initial policy15 publication or a new step16 update**. The direct error is a Unix-domain IPC path failure, rather than an observed CUDA OOM or TCP port collision.

The primary fix removes only the three temporary-directory variables from the **remote forwarding allowlist**, allowing workers to inherit their node holders' short private IPC paths. It does not clear the coordinator's own temporary-directory settings.

### Recorded evidence

| Evidence | Recorded outcome and scope |
| --- | --- |
| Original run | Contiguous real traceback; 150-byte endpoint exceeds 107-character ZMQ limit |
| Historical CPU regressions | 45 tests passed; original test log and acceptance record included |
| Original-cluster CPU ZMQ checks | Actual bind passed on all eight nodes; endpoint lengths 73–74 bytes |
| Subsequent GPU recovery | 16 training ranks restored parameters, Adam counters and RNG from CP15 |
| Initial publication after recovery | 12 TP4 inference replicas installed policy15; recorded synchronization 54.602446810982656 s |
| Actual subsequent updates | Steps16/17 updated, saved checkpoints and published policies16/17; each consumed 16 groups × 8 = 128 nonpadding trajectories, with 0 synthetic padding |
| Original metrics | Two exact JSONL rows for steps16/17, 98 tags per row, with original acceptance hashes |
| Supplemental portable checks | Standard-library CPU Unix-socket probe and offline source/patch/evidence/hash verifier passed; neither launches RL |

Hardware was 8 nodes × 8 H20, split into 2 trainer nodes (world size16) and 6 rollout nodes (12 TP4 replicas), using GRPO/DAPO, Score Centering, FSDP2/SP8 and separate_async. The model and recorded source revisions are in the package's environment note.

A separate **associated recovery fix** changes the supervisor from a legacy constructed launch path to the attempt's actual `launch.json`, with ticket, starting-step and source-acceptance-hash checks. It is documented separately and is not offered as a second independently GPU-reproduced benchmark case.

One failed attempt and one subsequent recovery are engineering evidence, not a multi-seed performance study. Exported reward/log-probability metrics are not claimed as fixed LIBERO evaluation success rates or a measured learning improvement.

### Material supplied

The complete sanitized review package has **39 UTF-8 files**, including:

- case description, experiment settings and limitations;
- before/after source snapshots for two scheduler modules and separate minimal patches;
- original traceback, historical CPU log, selected eight-node and GPU restore/update/publication receipts;
- unchanged original metric rows for steps16/17;
- source-to-export hashes, length-preserving sanitization notes and a sealed per-file SHA-256 manifest;
- two offline CPU inspection utilities and their recorded results.

The original local ZIP is 78,687 bytes, SHA-256:

```text
8d9c0ab9c7a4b12c2c2efd4447e161afdda4efdf6ebddba25d795613fab847f7
```

The original ZIP is included at [attachments/AutoRLBench-IPC-review-v1.zip](attachments/AutoRLBench-IPC-review-v1.zip). The 39 original files are also expanded here, unchanged, so maintainers can read the case and verify its sealed MANIFEST directly. No weights, checkpoints, credentials or private service endpoints are included.

### Open requirements and questions

I have read [the contribution workflow](https://thinking-space.github.io/AutoRLBench/contribute.html#submit), `CONTRIBUTING.md` and the validation requirements. I have **not** yet:

- reproduced the faulty full experiment in an approved digest-pinned shared image;
- provided a complete self-contained RL workspace, portable CP15 reconstruction/source, asset `prepare.sh`, `task.toml`, `run_spec.json`, oracle and verifier;
- run the official validator/startup rerun or calibrated a fair Solution threshold.

The Python socket probe demonstrates only the OS-level mechanism. Original-cluster recovery does not substitute for approved-image evidence. Local scheduler snapshots are supplied for technical review; publication/contribution licensing still needs to be agreed while preserving upstream licenses.

Would this pre-first-new-update initialization failure be suitable as an Issue-tier infrastructure case? In particular:

1. How should the generic startup-rerun requirement of at least two real training steps apply when the intended fault prevents initialization itself?
2. What reduced but complete shared-image RL replay would you recommend for the multi-node/CP15 dependency while retaining this same failure cause?

This draft proposal supplies raw material for maintainer review. It adds no task under tasks/ and is not a benchmark-ready task PR.
