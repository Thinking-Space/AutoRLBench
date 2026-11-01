# Missing difficult examples inflate training reward

This task begins with a materialized data incident and preserved native actor and critic checkpoints at step 3. The harness consumes the submitted `workspace/loader.py`, restores the saved training state, and continues rollouts 4–7. Setup does not train a prefix or reconstruct an incident from an agent's output.

| Item | Contract |
| --- | --- |
| Framework | miles / JustRL-II, PPO |
| Model | Public `openbmb/JustRL-II-base-model`, revision `b558bbc71d9433e7c5869effcbac07d4c7f2c5b4` |
| Framework source | JustRL-II revision `75814318f243af7e39af337c1173fc3927c9635b` |
| Dataset | Frozen public UltraData-RL-2609 Math selection: 64 training and 16 fixed evaluation examples |
| Runtime | Pinned `justrl2-data-relocation` image; three physical GPUs: actor 1, critic 1, inference 1; TP1/CP1/DP1 |
| Submission | `loader.py` with `resolve(root, record) -> relative_path`, plus `report.md` |
| Continuation | Checkpoint iteration 3; rollouts 4–7; total schedule remains 8 |
| Current acceptance status | Packaged GPU run 1430440 passed the hidden fresh verifier with score 1; see [validation](environment/validation.json) |

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
python3 -m autorlbench harbor-shell tasks/data/justrl2_data_relocation \
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

## Runtime and evaluation boundaries

`environment/incident_engine` contains an explicit runtime module list and the licensed compatibility/checkpoint helpers. It excludes tests, `build.py`, `prefix.py`, and `reference_loader.py`. The maintainer reference resolver exists only under `evaluations/solution/oracle/`; neither setup nor the run specification installs it into the agent workspace or prepared engine.

The editable workspace starts with the defective resolver and historical observations. Instructions remain neutral about the diagnosis and repair algorithm. The canonical issue-evaluation instruction, judge, prompt, and shell scaffold are unchanged. The solution harness checks the report submission, then runs the immutable prepared driver; it does not execute a modified workspace launcher. Only the submitted resolver is copied into the new run.

The trusted driver keeps checkpoint selection, original logical sampling order, generation, rewards, training, and evidence collection outside the submitted resolver. It independently opens the returned file and checks the requested ID and exact content. A resolver may retry transient reads; substituting a different example is not accepted.

The hidden grader uses a separate copy of the frozen independent auditor, SHA-256 `db21b6dd204f55efb4e4ce06aa65dfb0a7dd76e9ac35caa87fc31223cbfe4286`. It receives the prepared scene separately and always uses fresh-output verification. Passing requires:

- Exact scheduled logical sample groups and original content for rollouts 4–7, with complete read, terminal, admission, and available attempt accounting.
- Exact native actor/critic model and Adam readback, scheduler and all RNG-family restoration, native iteration 3, Adam counter 4, and scheduler progress 64.
- Four successful real optimizer updates per role, counters 5–8, and iteration-7 checkpoint readback with step delta 4.
- Rehashed final checkpoint payloads, unchanged sealed scene inputs, and the complete fixed 16-example final evaluation.
- Scene-bound model file inventory when prepared HF assets move to the new runtime path. The snapshot contains seven measured `model_files` pins from the verified metadata capture; no pins are invented here.

Reward improvement, repaired-model capability, zero transient retries, and a fixed easy/hard percentage are not pass thresholds. The wrapper emits an integer binary score and retains the auditor's boolean score and detailed proof in `metrics`.

## Evidence status and provenance

Earlier eight-update from-base experiments and their successful repaired-run grading remain historical evidence of the data defect. They do not prove continuation from this incident checkpoint. Their original validation receipt and old source manifest are preserved byte-for-byte under `environment/history/`; historical oracle comparison/grade receipts are under `evaluations/solution/oracle/history/`. Workspace `logs/recorded/` is labeled accordingly.

[Packaged run 1430440](environment/receipts/packaged-reference-grade.json) passed the exact prepared-engine run specification and hidden verifier: 64 generated training samples, four updates per role, both final checkpoint payloads rehashed, seven relocated model-asset files rehashed, and all 16 fixed evaluation examples complete. Its observed final evaluation reward is 0.25; reward quality is not a pass threshold. [Preparation 1430414](environment/receipts/prepare-validation.json) installed the original scene before solver entry without training.

[Reference run 1430316 and independent audit 1430462](../../../proposals/justrl2_data_relocation/incident/evidence/README.md) provide separate measured restore events, raw-source hashes and standalone curves. The curves describe that reference run, not the packaged run.

`environment/source_manifest.json` inventories every current task file except itself. `environment/validation.json` records CPU checks, native reference acceptance, actual installation and packaged acceptance separately. All task snapshots retain the same scene and seven model-file pins. Informational validation fields were updated after the run; executed code and input pins are unchanged. The immutable `SCENE.json` remains the original assembly receipt; later restoration proofs are recorded outside it. Refresh the manifest after any further metadata additions.

No LLM blind tests or agent evaluations were run. Synthetic fixtures cover CPU/schema behavior; the recorded GPU runs establish restoration and continuation acceptance. No agent success rate or model-quality advantage is claimed.
