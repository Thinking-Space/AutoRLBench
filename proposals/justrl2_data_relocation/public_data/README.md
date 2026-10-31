# Public reconstruction of the frozen math store

This standard-library-only preparer rebuilds the selected 80 prompt/answer pairs
from `openbmb/UltraData-RL-2609`, config `Math`, split `train`, pinned to commit
`e6ecfa733708a4c54b5a98c3ca0fd16fc6923790`.

The 64 training backing files, `manifest.json`, and `eval.jsonl` reproduce the
original **66 runtime files byte for byte**. `CASE.json` is newly generated public
provenance. Its hash is deliberately different from the original CASE hash.
`PUBLIC_PROVENANCE.json` binds the public assets, preparation inputs, and verified
runtime hashes. The output store therefore contains 68 files altogether.

## Preparation interface

From the repository root, with the four shards already downloaded:

```bash
python3 -B proposals/justrl2_data_relocation/public_data/prepare_public.py \
  --source-dir /tmp/ultradata-math \
  --out /tmp/relocation-store
```

To fetch missing shards from the pinned public revision:

```bash
python3 -B proposals/justrl2_data_relocation/public_data/prepare_public.py \
  --source-dir /tmp/ultradata-math \
  --out /tmp/relocation-store \
  --download
```

`--source-dir` directly contains `Math_part-1-of-4.jsonl` through
`Math_part-4-of-4.jsonl`; it is the shard directory, not its repository parent.
`--out` must be absent or empty. Existing shards must match their pinned sizes and
SHA256 values, including when `--download` is supplied. A corrupt existing shard
is rejected instead of silently replaced. Network access is opt-in and downloads
only these four Math files. No fixture content is sent to the server.

By default, `selection.json` and `sources.lock.json` are loaded beside the script,
independent of the working directory. `--selection PATH` and `--lock PATH` permit
explicit input files. A successful CLI invocation prints one JSON summary with
counts and the source, manifest, eval, new CASE, and receipt hashes.

The callable interface is:

```python
prepare(
    source_dir,
    out,
    *,
    selection_path=DEFAULT_SELECTION,
    lock_path=DEFAULT_LOCK,
    download=False,
)
```

It accepts string or `Path` arguments, returns the receipt dictionary, and raises
`ValueError` for invalid identities/counts/hashes or an `OSError` subclass for
missing inputs and nonempty outputs. It does not launch training or modify a
previous store.

## Frozen membership and serialization

`selection.json` contains public UUIDs and offsets, unchanged benchmark IDs,
train/eval order, canonical prompt/label/content hashes, raw public record hashes,
and expected SHA256 values for all 66 runtime files. It contains no question or
answer text, private source path, private source-file hash, or private source-line
hash. A public record hash covers its complete raw JSONL line, including its line
terminator.

The selected historical accuracy numerics and easy/hard assignments are **frozen
author-supplied historical strata, not public dataset measurements**. They were
retained under author authorization solely for runtime byte identity. They are
not current-model results or newly measured performance. The public corpus does
not contain those accuracy measurements. Selection uses the published frozen ID
list; it does not recreate a historical difficulty-filtering procedure.

| Split | Easy | Hard | Total |
|---|---:|---:|---:|
| Train | 32 | 32 | 64 |
| Eval | 8 | 8 | 16 |

Each runtime record has exactly the original `id`, `prompt`, `label`, and
`metadata` fields. Metadata retains only `difficulty`, `source_line`, and
`historical_accuracy`. `source_line` is also verified against the selected public
UUID's one-based global position across the four shards. Questions and answers
are copied without suffix changes, whitespace changes, or mathematical
normalization.

Backings and manifest use:

```python
(json.dumps(value, ensure_ascii=False, sort_keys=True,
            indent=2, allow_nan=False) + "\n").encode("utf-8")
```

Each eval row instead uses the original single-line representation:

```python
(json.dumps(value, ensure_ascii=False, sort_keys=True,
            allow_nan=False) + "\n").encode("utf-8")
```

Eval rows and manifest records preserve the selection order. Canonical content
hashes use `ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, and
`allow_nan=False`, encoded as UTF-8. All 240 selected prompt/label/content hashes
and all 66 runtime file hashes must pass before output files are created.

## Public CASE and receipt

The new CASE retains `schema_version: 1`, `synthetic_fault_injection: true`, the
original IDs/splits/source-line integers and prompt/label/content hashes. Each
CASE row names its public UUID, shard, zero-based row offset, and **public** raw
record hash. `difficulty_basis` describes the frozen author labels. No private
source path, source-file hash, or original source-line hash is copied.

`source.sha256` is a deterministic **composite public source manifest digest**.
It is not the digest of an old private source file. Its exact calculation is:

```python
public_manifest = {
    "repo": source["repo"],
    "revision": source["revision"],
    "config": source["config"],
    "split": source["split"],
    "files": source["files"],
}
# files is ordered shard 1..4; each entry has path, size, sha256, rows.
source_sha256 = hashlib.sha256(
    json.dumps(public_manifest, ensure_ascii=False, sort_keys=True,
               separators=(",", ":"), allow_nan=False).encode("utf-8")
).hexdigest()
```

`source.fileshashes` additionally maps each public repository file path to its
SHA256. `source.sha256_meaning` states the composite definition. The `sha256`,
`sha256_meaning`, and `fileshashes` fields are not included in the composite input.

The receipt records that same public source manifest; hashes of the preparer,
selection, and lock; all runtime file hashes and sizes; verification counts; and
the **new** CASE hash. No timestamp or machine path enters the receipt. The
receipt hash changes if the preparer bytes change, even when the runtime bytes
remain identical.

The preparer validates everything before creating runtime output and rechecks
the source shards immediately before writing. `PUBLIC_PROVENANCE.json` is
written last as the completion receipt. An interrupted filesystem write can
leave an incomplete directory; repeated preparation never overwrites a nonempty
directory.

Runtime packaging should consume the new CASE/receipt and record the new CASE
hash in fresh evidence inputs. Existing consumers that require a 64-character
`source.sha256` receive the documented composite digest. Do not reuse an old CASE
hash or relabel old private lineage as public lineage. Runtime equivalence is
separate from claims about a new model run.

## Pinned public assets

The source lock contains sizes, row counts and SHA256 values. The four files have
32,412 records in total; all are verified before selecting the frozen 80 rows.

| Public repository path | Rows | Bytes | SHA256 |
|---|---:|---:|---|
| `data/Math/Math_part-1-of-4.jsonl` | 10000 | 4803619 | `7253d74d827b0c357fef12646a4c247b6894e95adafd56cdae67b713d4220150` |
| `data/Math/Math_part-2-of-4.jsonl` | 10000 | 4182093 | `33a2aaca39443504dff81e234421308ebbb3135627a1dfa5501e3487f4b69f7f` |
| `data/Math/Math_part-3-of-4.jsonl` | 10000 | 4387044 | `778ecc93e229f2a1c3197561c977e87f86afe630383bf1b3779adf05c07e5d70` |
| `data/Math/Math_part-4-of-4.jsonl` | 2412 | 1233160 | `1838dd71bf537ff8a5d0ed87aa11b43d4ef07220a21b5a759a0d74b57f0e289e` |

The downloader resolves each path at the full commit above, using this template:

```text
https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}
```

## Dataset card and licensing record

The pinned UltraData card declares `apache-2.0`; its LICENSE is Apache 2.0 and
names OpenBMB. The card also states that upstream MIT, CC BY 4.0 and CC BY-SA 4.0
terms remain applicable, and includes a separate restriction on unchanged
third-party redistribution, rehosting, or commercial repackaging without prior
written permission. These statements are recorded together without legal
interpretation. This directory distributes preparation code, IDs and hashes;
corpus files are fetched from the public source.

The card identifies DAPO-Math-17k, DeepScaleR-Preview-Dataset and DeepMath-103K as
Math inputs. Their inspected cards declare Apache-2.0, MIT and MIT respectively;
this is card-level metadata, not a verified per-row licensing chain. Runtime
preparation uses UltraData alone, with exact question/answer membership verified
there.

Pinned primary documents:

```text
https://huggingface.co/datasets/openbmb/UltraData-RL-2609/blob/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/README.md
https://huggingface.co/datasets/openbmb/UltraData-RL-2609/blob/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790/LICENSE
https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/blob/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29/README.md
https://huggingface.co/datasets/zwhe99/DeepMath-103K/blob/5cf055d1fe3d7a2eb19719ac020211469736ae44/README.md
https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/65877096c24ffa7abc4e4fa5edb95cf3413a5674/README.md
```

The inspected UltraData README SHA256 is
`c9e13e938e28adeb3e5529f5c78b2f81d060044eb6d645964dc2dcdae910e556`;
LICENSE SHA256 is
`202d02b14796563546e1a1d1e761410fa9e399bf7e1b85d8b43c972658341202`.
These documentation pins are recorded in `sources.lock.json`; `--download`
fetches only the four data shards.

## Validation

```bash
python3 -B -m unittest discover \
  -s proposals/justrl2_data_relocation/public_data \
  -p test_prepare_public.py -v
```

The CPU-only suite uses four tiny synthetic shards. It checks exact runtime
serialization, public CASE/receipt bindings, all content and runtime hashes,
source size/digest/count changes, missing data, duplicate or insufficient
selections, frozen numeric metadata, default CLI paths, optional pinned
downloads, and rejection of repeated output. Download unit tests substitute
in-memory responses; they do not contact the network.

Verified on 2026-10-31: **23 tests passed**. A separate real-data smoke test
downloaded all four pinned shards into a fresh temporary directory using
`--download`, reconstructed the store, and compared all 66 runtime files directly
with the original fixture. All bytes matched; all 240 content hashes passed; the
existing `BackingStore` read all 64 backings successfully. Repeated output was
rejected. The original 67 files, including the original CASE, remained unchanged.
The smoke output was temporary; no model run was performed.

| Artifact | SHA256 |
|---|---|
| Composite public source manifest | `12be8265c79ac94900eba6897a4c90fb28378b83f78a2fba8fdd60b44a4a916d` |
| Runtime `manifest.json` | `cc9a936f9ba63d10c34c34451a721d5693e0e9b415a05852f553741bcee122ee` |
| Runtime `eval.jsonl` | `1c5e50bdb78128ef000023ac18874939482f8b097613290265fd8f3984e8b0aa` |
| New public `CASE.json` | `7957063f1f6705aa7c7167dff93778cb44998d4d594704977fcb66af08cbb7d1` |
| `PUBLIC_PROVENANCE.json` | `0f55b4cafb7095650325b88ac94138481066ce681835c67ad045847bc20dcd78` |
| Canonical 66-entry runtime SHA256 map | `1314e0f242ae9510ce01e6fc603ce6aa3f2fb419b6374ecf47dfe5c4df0c6c87` |
| `selection.json` | `09f9db60274c3405f744f06e61feaacfd18720709faf14a48a9999686deb127d` |
| `sources.lock.json` | `70043559f0bf7cf441868c10d7aaec7db91b2ccf74d7bcc0ac7b89f5914b82f2` |
| `prepare_public.py` | `3f17ebffc0b231dc9ab989e8eddf05f769e0a5352674fe0a841c6f09cfb0d189` |

The runtime-map digest uses the same canonical JSON serialization described
above, applied to the receipt's `runtime_sha256` dictionary. The full per-backing
hash list is present in both the selection and receipt.
