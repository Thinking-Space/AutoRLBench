#!/usr/bin/env bash
# Sanitized replay record: site account and filesystem paths were removed.
#SBATCH --job-name=autorl-slime-import
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=import_shared_slime_%j.out

set -euo pipefail
: "${AUTORLBENCH_RAW_ROOT:?set to the raw contribution root}"
OUT="$AUTORLBENCH_RAW_ROOT/images/slime-v0.3.2-cu129.sqsh"
mkdir -p "$(dirname "$OUT")"
if [[ -s "$OUT" ]]; then
    echo "IMAGE_EXISTS $OUT"
else
    enroot import -o "$OUT" docker://slimerl/slime:v0.3.2-cu129
fi
sha256sum "$OUT"

ROOT="$AUTORLBENCH_RAW_ROOT/router-bias"
for arm in buggy fixed; do
    rm -rf "$ROOT/${arm}_workspace/official_outputs"
    srun \
        --container-image="$OUT" \
        --container-mounts="$AUTORLBENCH_RAW_ROOT:$AUTORLBENCH_RAW_ROOT" \
        --container-workdir="$ROOT/${arm}_workspace" \
        bash -lc "TRAINING_RUN_DIR=official_outputs python3 train.py" \
        > "$ROOT/${arm}_workspace/official_run.log" 2>&1
done
