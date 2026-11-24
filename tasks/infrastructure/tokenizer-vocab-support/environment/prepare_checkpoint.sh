#!/usr/bin/env bash
set -euo pipefail

environment=${1:?expected the task environment directory}
assets=${2:?expected a review-accessible checkpoint asset manifest}
: "${AUTORLBENCH_PREPARED:?prepare requires the harness prepared path}"
: "${AUTORLBENCH_WORKSPACE:?prepare requires the harness workspace path}"

native_manifest="$AUTORLBENCH_WORKSPACE/scripts/cp725.sha256"
destination="$AUTORLBENCH_PREPARED/global_step_725"
printf '%s  %s\n' fab5f6b531995601b7bbd678c21b1d95d0e1cd0c437fcb47135805edda68f079 "$native_manifest" |
    sha256sum --strict --check -

if [[ -d "$destination" ]]; then
    (cd "$destination" && sha256sum --strict --check "$native_manifest")
    exit 0
fi

staging=$(mktemp -d "$AUTORLBENCH_PREPARED/cp725.XXXXXXXX")
trap 'rm -rf -- "$staging"' EXIT
python3 "$environment/prepare_fetch.py" --manifest "$assets" --dest "$staging/download"
checkpoint="$staging/download/global_step_725"
(cd "$checkpoint" && sha256sum --strict --check "$native_manifest")
mv -T -- "$checkpoint" "$destination"
