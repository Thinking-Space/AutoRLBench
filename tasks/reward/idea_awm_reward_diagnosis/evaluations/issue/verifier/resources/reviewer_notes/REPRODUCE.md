# Environment and reproduction

This issue-only task uses `idea-issue` for CPU diagnosis and offline checks.
The optional `idea-historical` image is **not yet published**; no GPU training rerun
or corrected control is supplied.

## Shared images and checks

Image definitions live in the benchmark's top-level `docker/images/`, with pinned
requirements embedded directly in each Dockerfile. Include both IDEA Dockerfiles
in the PR. No separate `docker/idea-runtime/` requirements files, task checkout,
host conda environment or model files are required to build the images. Every
`docker/images/*.Dockerfile` carries an identical standard tooling block (the
activate hook and fetch tooling install paths it declares), and the fetch
helper ships per task in `environment/`. Network access is still needed for
base images and package sources. The task's
`environment/runtime/` files are provenance snapshots, not Docker build inputs.
For this task build manually from the benchmark root and tag as the
Dockerfile header declares:

```bash
docker build -f docker/images/idea-issue.Dockerfile docker/
docker tag <image-id> autorlbench/idea-issue:py312-20261023-deps1
```

The static image supplies pinned W&B/protobuf, OmegaConf and PyArrow. The networked
prepare stage downloads eight files from `locaszzz/AgentWorldModel-1K` at commit
`6f8a98a043c631ba9c88dd1d1d1d1829a1f9dca5` and checks their original hashes
and sizes. Run the following from a prepared task root, not the data-free submission:

```bash
docker run --rm --network none --read-only --tmpfs /tmp:rw,exec \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/task:ro" -w /task \
  autorlbench/idea-issue:py312-20261023-deps1 \
  python3 -B -m unittest discover -s evaluations/issue/verifier/tests -v
```

For current prepare/agent-isolation/verifier-transport wiring, use Python 3.11+
and explicit temporary output paths outside the task:

```bash
python3 -B evaluations/issue/verifier/resources/tools/validate_container.py \
  --benchmark-root /path/to/AutoRLBench \
  --work-dir /path/to/temporary-run --report /path/to/temporary-report.json
```

This uses a loopback fixture, not a semantic judge or training run.

## Optional historical training

The historical image defines CUDA 12.8.1, separate Python 3.12.0 (IDEA) and
3.12.13 (AWM) prefixes, torch 2.8.0, vLLM 0.11.0 and captured dependency pins.
Build it explicitly from the benchmark
root's docker/ context and tag as declared in the Dockerfile header:
`docker build -f docker/images/idea-historical.Dockerfile docker/`.
No installed host conda environment is consulted.

Supply the exact initialization checkpoint separately. The model revision is
`61a407b4fefb501bfdeaef659de2cd8bbca05536`; the dataset revision is
`e49176428defd16416d9f5b11ab7eaca170d5732`. A pinned Hub revision is not independent
attestation that uploaded weights match the historical local checkpoint.
After the image is built, from the task root:

```bash
docker run --rm --network none --gpus all --shm-size 32g \
  -v "$PWD/workspace:/workspace:ro" \
  -v /absolute/path/to/exact-checkpoint:/model:ro \
  -v /absolute/path/to/new-output-directory:/results \
  autorlbench/idea-historical:py3120-awm31213-torch280-cu128-20261023 \
  bash /workspace/run_local.sh --train --model-path /model --output /results
```

This configuration requires eight 80GB GPUs. Use `--smoke --config-only` instead
of `--train` for input, MCP and Hydra checks without inference or optimization.
Inputs must already be downloaded by prepare; output goes to `/results`. Explicit interpreter/model overrides
are supported by `workspace/reproduce.py`; do not submit credentials or
`runtime.local.json`.
