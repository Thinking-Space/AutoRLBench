# Review the current math training experiment

This is one half of a paired assessment task. Agents receive the same neutral
instruction, historical observations, sealed step-3 checkpoint and dataset.
They decide whether the current implementation needs intervention. A healthy
workspace may be submitted without code changes; an unhealthy workspace needs
a justified repair. The report describes the starting implementation.

| Case (maintainer information) | Starting implementation | Reference action |
| --- | --- | --- |
| `justrl2_data_relocation` | Uses stale manifest paths | Report unhealthy and repair resolution |
| `justrl2_data_review_control` | Resolves current paths by ID/content | Report healthy; leave code unchanged |

This package's hidden starting verdict is **healthy**. This README and the
case-specific provenance are maintainer material, outside the normal agent
mounts. Deploy the two cases separately so an agent does not inspect its paired
answer. Case names, expected verdicts and maintainer success records are not
part of the answer workspace. The shared sealed scene records the historical
migration in both cases; the current loader determines whether it remains a
problem.

## Why assess before changing code?

In the historical fault, migrated difficult examples could not be read. The
collector skipped failed reads and sampled replacements. Training continued,
and reward among admitted samples rose even though the training population had
changed. Those logs justify investigation, but a current resolver may already
handle the migrated files. An agent must inspect present behavior instead of
assuming every task needs a patch.

## Submission and acceptance

For the solution evaluation, `report.md` starts with `Verdict: healthy` or
`Verdict: unhealthy`, followed by the evidence, checks and chosen action. It
assesses the starting implementation, even if the agent subsequently repairs
it. Only `loader.py` is executed from the submitted code; the interface boundary
does not assert that it is broken.

The hidden grader requires both the correct assessment and the unchanged fresh
continuation contract: original logical IDs/content, complete checkpoint state
restoration, four actual updates per role, final checkpoint verification and
fixed evaluation completion. Declaring healthy does not skip runtime checks.
No patch is required for healthy code; harmless changes are judged by behavior.
Reward improvement and a fixed difficulty percentage are not pass thresholds.

Issue evaluation asks only for the current-implementation assessment and its
evidence. It uses the same neutral instruction in both cases and a hidden oracle
specific to the starting code.

## Install the pinned local scene before agent entry


The supported distribution mode is a **pinned local scene directory** mounted through the stock harness dataset cache. Copy the complete verified scene into the host datasets directory at:

```text
<datasets-dir>/incidents/b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac/SCENE.json
```

The directory must also contain the real step-3 actor/critic checkpoints, datasource state, incident dataset, and provenance. Use a new destination directory and preserve the complete artifact; the cache directory name does not replace receipt and payload verification.

For example, an operator can copy the artifact and open the prepared runtime without starting an agent:

```bash
incident_datasets=/absolute/path/to/benchmark-datasets
incident_sha=b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac
mkdir -p "$incident_datasets/incidents/$incident_sha"
cp -a /path/to/verified-incident-scene/. "$incident_datasets/incidents/$incident_sha/"
python3 -m autorlbench harbor-shell tasks/data/justrl2_data_review_control \
  --datasets-dir "$incident_datasets"
```

Pass the same `--datasets-dir` on stock Docker or Harbor evaluation commands. The harness mounts that directory and exports its container path as `AUTORLBENCH_DATASETS`; setup selects `$AUTORLBENCH_DATASETS/incidents/<scene_sha256>` automatically. No additional harness flag or arbitrary host environment forwarding is needed.

An explicit `RELOCATION_SCENE_SOURCE` **inside the runtime environment** takes precedence over the cache. This remains available for operator-managed preparation:

```bash
export RELOCATION_SCENE_SOURCE=/mounted/pinned-incident-scene
bash "$AUTORLBENCH_ENVIRONMENT/prepare.sh"
```

The source-selection order is explicit environment path, pinned datasets-cache directory, then an optional pinned archive URL. An invalid explicit path or mismatching cached scene fails verification rather than silently selecting another artifact. The current snapshot declares no public scene URL; a source URL is not required for local-artifact mode. Internal scene metadata and tensor files are not uploaded by this package.

Preparation verifies the pinned public model, tokenizer, framework source and data inputs, then performs the existing exact CPU conversion to `$AUTORLBENCH_PREPARED/assets/hf-safetensors`. This is asset conversion, not model training. It installs the trusted engine into `$AUTORLBENCH_PREPARED/incident_engine`, copies and verifies the supplied scene into `$AUTORLBENCH_PREPARED/incident`, and exposes the workspace `incident/` link before the agent starts. The receipt pin and complete scene inventory are checked; an absent or mismatching scene fails preparation. No fresh-model fallback or replacement prefix is generated.

The prepared directory is owned and protected by the harness. The complete scene is approximately 66.7 GB; allow separate storage for its installation, public model assets, and fresh output checkpoints. `materialize.py` also supports an archive only when its hash and size are explicitly pinned; the current snapshot provides the local-directory mode.

## Visibility and provenance

`environment/` contains only the shared preparation code, frozen runtime and
case-independent scene/model pins. It is byte-identical across the two cases.
The workspace differs only in the current `loader.py`; both use identical
historical observations. Reference code and expected verdicts are under hidden
`evaluations/`; author receipts and source inventories are under `provenance/`,
which is not mounted by the normal Docker/Harbor agent stage.

The inspection boundary does not make the two implementations indistinguishable:
their code and read behavior are precisely the evidence an agent should inspect.
It removes task-author statements that dictate the answer. The local development
backend provides no file isolation and is not an agent-blinding environment.

## Validation scope

The unchanged native continuation was measured in packaged GPU run 1430440 and
independent reference audit 1430462. Its original receipts remain intact under
[provenance/receipts/](provenance/receipts/). The healthy loader has the same
executable AST as the previously GPU-validated reference resolver; only its
maintainer-only docstring was replaced by a neutral one.

The new assessment gate and healthy/no-change submission path are checked with
explicit synthetic continuation fixtures and real dataset content reads.
[validation.json](provenance/validation.json) distinguishes these checks from
inherited GPU evidence. No new LLM agent evaluation or blind test was run, and
no agent success rate is claimed. Existing GPU receipts do not by themselves
prove a new assessment report or a separately executed healthy-agent run.

[source_manifest.json](provenance/source_manifest.json) inventories this package
and its source adaptations. Raw checkpoint bytes and their scene hash are
unchanged; their original assembly receipt is not rewritten for this protocol.
