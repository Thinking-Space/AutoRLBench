#!/usr/bin/env bash
# Fill the shared download cache by running every task's prepare.sh once.
#
# ---------------------------------------------------------------------------
# What this is for
# ---------------------------------------------------------------------------
# Moving the downloads out of the image build (where they used to sit, one
# `RUN curl` per task Dockerfile) and into a prepare stage means the evaluation
# host needs network access. That is a bad trade on a cluster where the GPU
# nodes are airgapped, so this script buys it back: run it once on a machine
# that *does* have network, and every prepare stage afterwards is a cache hit.
#
#   ./docker/warm-cache.sh                        # fills <repo>/models and <repo>/datasets
#   autorlbench evaluate ...                         # offline, all hits
#
# Point it somewhere else when the cache is meant to be shared over NFS:
#
#   ./docker/warm-cache.sh --models-dir /mnt/shared/models \
#                          --datasets-dir /mnt/shared/datasets
#   autorlbench evaluate --models-dir /mnt/shared/models \
#                     --datasets-dir /mnt/shared/datasets ...
#
# It is also just a fast way to find out whether a newly written prepare.sh
# works, without waiting for an agent run to get that far.
#
# ---------------------------------------------------------------------------
# What it does NOT do
# ---------------------------------------------------------------------------
# It does not produce a runnable workspace. Each task gets a throwaway
# /workspace and /autorlbench/prepared that are deleted when it finishes -- only
# the two cache roots survive, which is the whole point. A real evaluation
# still runs prepare.sh; it just finds everything already downloaded.
#
# Tasks without a prepare.sh are skipped: they have nothing to download.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
    cat >&2 <<'EOF'
usage: docker/warm-cache.sh [--models-dir DIR] [--datasets-dir DIR] [task-id ...]

  --models-dir DIR    host directory for model downloads; created if missing
                      (default <repo>/models, or $AUTORLBENCH_MODELS_DIR)
  --datasets-dir DIR  host directory for dataset downloads; created if missing
                      (default <repo>/datasets, or $AUTORLBENCH_DATASETS_DIR)
  [task-id]           warm only these tasks (directory names under tasks/);
                      default is every task that has a prepare.sh

Each task runs in its environment image (the tag in the `# autorlbench-image:` header
of docker/images/<environment.image>.Dockerfile), built on first use exactly as
`autorlbench build-images` would.

env:
  AUTORLBENCH_HF_ENDPOINT passed through as HF_ENDPOINT, for a mirror
EOF
    exit 2
}

models_dir="${AUTORLBENCH_MODELS_DIR:-$repo_root/models}"
datasets_dir="${AUTORLBENCH_DATASETS_DIR:-$repo_root/datasets}"
while [ $# -gt 0 ]; do
    case "$1" in
        --models-dir)   [ $# -ge 2 ] || usage; models_dir="$2"; shift 2 ;;
        --datasets-dir) [ $# -ge 2 ] || usage; datasets_dir="$2"; shift 2 ;;
        -h|--help)      usage ;;
        --*)            echo "error: unknown flag $1" >&2; usage ;;
        *)              break ;;
    esac
done
mkdir -p "$models_dir" "$datasets_dir"
models_dir="$(cd "$models_dir" && pwd)"
datasets_dir="$(cd "$datasets_dir" && pwd)"

# The task list comes from the same reader the evaluator uses, so "which task
# runs in which image" has one answer rather than a shell glob's approximation
# of one. Output is `<task-dir>\t<environment>\t<image-tag>\t<prepare-script>`.
mapfile -t entries < <(python3 - "$@" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from autorlbench.core.types import DEFAULT_PREPARE, EnvironmentImage  # noqa: E402
import tomllib  # noqa: E402

wanted = set(sys.argv[1:])
found = set()
for metadata_path in sorted(Path("tasks").rglob("task.toml")):
    if "raw" in metadata_path.relative_to("tasks").parts:
        continue
    environment = tomllib.loads(metadata_path.read_text(encoding="utf-8")).get("environment")
    if not isinstance(environment, dict) or not environment.get("image"):
        continue
    task_dir = metadata_path.parent
    if wanted and task_dir.name not in wanted:
        continue
    declared = environment.get("prepare")
    script = task_dir / str(declared or DEFAULT_PREPARE)
    if not script.is_file():
        if declared:
            print(f"error: {task_dir}: {script} does not exist", file=sys.stderr)
            raise SystemExit(1)
        continue
    found.add(task_dir.name)
    # The container sees environment/ mounted at /autorlbench/environment, so what
    # it needs is the path relative to that directory, not the host path.
    relative = script.relative_to(task_dir / "environment")
    image = EnvironmentImage(str(environment["image"]))
    print(f"{task_dir}\t{image.name}\t{image.tag}\t{relative.as_posix()}")

for name in sorted(wanted - found):
    print(f"error: no task named {name!r} has a prepare.sh", file=sys.stderr)
if wanted - found:
    raise SystemExit(1)
PY
)

if [ "${#entries[@]}" -eq 0 ]; then
    echo "no tasks to warm"
    exit 0
fi

echo "warming ${#entries[@]} task(s) into $models_dir and $datasets_dir"
echo

failed=()
for entry in "${entries[@]}"; do
    IFS=$'\t' read -r task_dir env_name image_tag script <<<"$entry"
    task_id="$(basename "$task_dir")"
    echo "=== $task_id  (image: $image_tag)"

    # Same "build only if missing" rule as `autorlbench evaluate`.
    if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
        if ! python3 -m autorlbench build-images "$env_name"; then
            echo "--- $task_id FAILED (image build)" >&2
            failed+=("$task_id")
            echo
            continue
        fi
    fi

    # Throwaway per task. prepare.sh writes weights into $AUTORLBENCH_WORKSPACE and
    # KEY=VALUE lines into $AUTORLBENCH_PREPARED/env; neither is what we are here
    # for, but both have to exist or the script fails on a missing directory.
    scratch="$(mktemp -d)"
    mkdir -p "$scratch/workspace" "$scratch/prepared"

    # Mirrors the prepare stage's mount table and environment -- see
    # DockerRuntime._mounts and SandboxRuntime.standard_env. Deliberately the
    # same, so that a prepare.sh which warms cleanly also runs cleanly.
    args=(
        docker run --rm
        -v "$scratch/workspace:/workspace"
        -v "$scratch/prepared:/autorlbench/prepared"
        -v "$models_dir:/autorlbench/models"
        -v "$datasets_dir:/autorlbench/datasets"
        -v "$repo_root/$task_dir/environment:/autorlbench/environment:ro"
        -w /workspace
        -e AUTORLBENCH_STAGE=prepare
        -e "AUTORLBENCH_TASK_ID=$task_id"
        -e AUTORLBENCH_EVALUATION_LEVEL=solution
        -e AUTORLBENCH_WORKSPACE=/workspace
        -e AUTORLBENCH_ENVIRONMENT=/autorlbench/environment
        -e AUTORLBENCH_PREPARED=/autorlbench/prepared
        -e AUTORLBENCH_MODELS=/autorlbench/models
        -e AUTORLBENCH_DATASETS=/autorlbench/datasets
        -e WANDB_MODE=offline
        -e WANDB_SILENT=true
    )
    [ -n "${AUTORLBENCH_HF_ENDPOINT:-}" ] && args+=(-e "HF_ENDPOINT=$AUTORLBENCH_HF_ENDPOINT")
    # GPUs stay opt-in here, unlike the real prepare stage, which reads the task
    # card's NVIDIA_VISIBLE_DEVICES. Warming is about the download cache, and a
    # prepare.sh that needs a GPU for a format conversion is expected to skip
    # that step when it cannot find one -- so this script keeps working on a
    # machine with no GPU at all, which is where caches usually get warmed.
    # Set AUTORLBENCH_DOCKER_GPUS (same spelling as the runtime honours) to also
    # warm whatever the conversion produces.
    if [ -n "${AUTORLBENCH_DOCKER_GPUS:-}" ]; then
        args+=(--gpus "$AUTORLBENCH_DOCKER_GPUS" "--shm-size=${AUTORLBENCH_DOCKER_SHM_SIZE:-16g}")
    fi
    # No offline switches and no --network flag: this is the one stage that is
    # supposed to reach the network, and reaching it is the entire job here.
    args+=(--entrypoint sh "$image_tag" -lc "bash /autorlbench/environment/$script")

    if "${args[@]}"; then
        echo "--- $task_id ok"
    else
        echo "--- $task_id FAILED" >&2
        failed+=("$task_id")
    fi
    rm -rf "$scratch"
    echo
done

echo "cache size: models $(du -sh "$models_dir" | cut -f1), datasets $(du -sh "$datasets_dir" | cut -f1)"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "all tasks warmed; evaluate with --models-dir $models_dir --datasets-dir $datasets_dir"
