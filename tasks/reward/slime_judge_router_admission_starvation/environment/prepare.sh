#!/usr/bin/env bash
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
# Offline historical diagnosis: no weights, datasets, or services are downloaded.
test -d "$AUTORLBENCH_WORKSPACE/observed_run"
