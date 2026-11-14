# Author-side data pipeline

These scripts are NOT part of the task lifecycle. They produce the committed
selection inputs the task ships with (`environment/selection/`) and the grader's
corrected-outputs table (`evaluations/solution/verifier/resources/`), from the
two public HF sources. Nothing here runs during prepare/agent/rerun/verifier
stages; `environment/prepare.sh` + `environment/assemble_dataset.py` are the
lifecycle side.

They live under `environment/` (agent-invisible) because their inputs, outputs
and comments spell out the task's answer.

## Pipeline

```text
BAAI/TACO ALL/train shards ─┐
                             ├─ build_dirty_pool.py ──> problems/<idx>.json + dirty_pool.jsonl
likaixin/TACO-verified ──────┘            (set difference; host, no execution)
                                              │
                                              v
                            purify_pool.py + exec_problem.py
                            (docker container, no network, executes the
                             problems' own bundled solutions under a loose
                             timeout through the task's own taco.py judge)
                                              │
                                              v
                   receipts.jsonl ──> dirty_manifest.jsonl  (committed, selection/)
                                     corrected_outputs.json (committed, verifier resources/)

BAAI/TACO ALL/test shard ──> build_dirty_pool.py --test-pool ──> problems_test/<idx>.json
                                                                     │
                                                                     v
                        purify_pool.py --passing-manifest ──> test_manifest.jsonl
                        (committed, selection/; verdict passes_loose =
                         the held-out split's own solutions pass its tests)
```

## 0. Sources

Download once into a work directory (about 4.6 GB):

```bash
mkdir -p <work>/sources && cd <work>/sources
for i in 0 1 2 3 4 5 6 7 8; do
  curl -L -O "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-0000${i}-of-00009.parquet"
done
curl -L -O "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/test-00000-of-00001.parquet"
curl -L -O "https://huggingface.co/datasets/likaixin/TACO-verified/resolve/main/taco_verified.json"
```

## 1. Candidate pools (host, pyarrow needed)

```bash
python3 environment/data_prep/build_dirty_pool.py \
    --sources <work>/sources --out <work>
python3 environment/data_prep/build_dirty_pool.py \
    --sources <work>/sources --out <work> --test-pool
```

Verifies the TACO-verified `id` <-> TACO train row mapping by question text
(the verified ids are 1-based positions in the concatenated shard order),
then writes one JSON per unverified train problem. Expect ~12.5k candidates
before difficulty filtering; only `stdin` problems with tests and runnable
solutions become per-problem files. `--test-pool` does the same marshalling
for the held-out split (no verified-set difference: that set covers train
rows only, so the test split's cleanliness is established in step 2 by
execution instead).

## 2. Purification (docker, no network)

Runs the candidates through the task's own judge under a loose per-test
timeout, and keeps only the problems where >= 2 textually distinct bundled
solutions fail the SAME test with IDENTICAL NON-EMPTY outputs (and no
solution passes): the wrong-expected-output family. The non-empty requirement
matters: when every solution produces an empty output the agreed "output"
establishes nothing about the expected output being wrong -- that pattern is
the unparseable-test-input family (GfG-style `N = 8\narr[] = {...}` test
inputs no stdin solution can read), and an empty-vs-empty match downstream
would vacuously confirm any print-nothing response. Everything else is
rejected with a reason (`passes_loose` = only timed out under the verified
set's stricter budget).

Any python image with numpy works; the local `verlai/verl` image is the
closest to the recorded stack:

```bash
docker run --rm --network none \
  -v <work>:/work \
  -v <repo>/raw_tasks/verl_taco_wrong_expected_output:/task:ro \
  <image> \
  python3 /task/environment/data_prep/purify_pool.py \
    --problems /work/problems \
    --receipts /work/receipts.jsonl \
    --manifest /work/dirty_manifest.jsonl \
    --corrected /work/corrected_outputs.json \
    --taco /task/workspace/verl/utils/reward_score/taco.py \
    --difficulties EASY,MEDIUM --workers 8
```

The test-split pool runs the same way with `--problems /work/problems_test`,
its own receipts file, throwaway manifest/corrected paths, and
`--passing-manifest /work/test_manifest.jsonl --difficulties EASY,MEDIUM`
(only the `passes_loose` receipts matter there; the split's EASY band alone
is too small for the 184-row held-out set; it finishes in minutes).

Resumable: receipts are appended per problem and already-tried problems are
skipped on restart. `--target-dirty N` stops once N dirty problems are
confirmed. Under the strict non-empty criterion the EASY/MEDIUM yield is
~10-18% of candidates, so run the pool to exhaustion (no target) and size the
mix from what survives. Wall clock is modest: most bundled solutions crash
fast on the corrupt test (the 1,514-problem first pass took 469 s with 8
workers).

## 3. Commit the outputs

```bash
cp <work>/dirty_manifest.jsonl <repo>/raw_tasks/verl_taco_wrong_expected_output/environment/selection/
cp <work>/test_manifest.jsonl  <repo>/raw_tasks/verl_taco_wrong_expected_output/environment/selection/
cp <work>/corrected_outputs.json <repo>/raw_tasks/verl_taco_wrong_expected_output/evaluations/solution/verifier/resources/
```

Before committing, shrink `corrected_outputs.json`: keep only `name`,
`fail_index`, `fail_input`, `corrected_output_sha256`, and `expected_sha256`
per entry, where the corrected digest is
`sha256(json.dumps([str(x) for x in corrected_output]))` and the expected
digest is `sha256(json.dumps(expected, ensure_ascii=False))` of the raw
dataset expected output (str or list form) at `fail_index` -- the grader
hashes the executed response's stdout lines the same way, and the expected
digest feeds the train-parquet tamper signature, so grading semantics are
identical while the file drops from ~53 MB to ~6 MB (full corrected outputs
stay in `receipts.jsonl` outside the repo).

Keep `receipts.jsonl` (the audit trail) outside the repo; record its sha256
and row count in the task README's assets table once the selection is frozen.

## 4. Calibrate the grader spec

After the buggy run and the clean control run exist:

* fill `required_final_score_max` in
  `evaluations/solution/verifier/resources/grader_spec.json` (between the
  control arm's rate ~0 and the defective arm's);
* fill `historical_final_baseline` with the defective arm's rate;
* pin the sha256s: assemble the parquets once (or run `prepare.sh`), then run

  ```bash
  python3 - <<'PY'
  import hashlib, json, pathlib
  root = pathlib.Path("raw_tasks/verl_taco_wrong_expected_output")
  spec_path = root / "evaluations/solution/verifier/resources/grader_spec.json"
  spec = json.loads(spec_path.read_text())
  for rel in spec["pinned_sha256"]:
      spec["pinned_sha256"][rel] = hashlib.sha256((root / "workspace" / rel).read_bytes()).hexdigest()
  spec_path.write_text(json.dumps(spec, indent=2) + "\n")
  PY
  ```

  (`taco_verl/test.parquet` only exists after a prepare run or a local
  `assemble_dataset.py` invocation; the launch script and taco.py can be
  pinned immediately).
