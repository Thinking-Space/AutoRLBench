#!/usr/bin/env bash
set -euo pipefail
cp "$(dirname "$0")/experiment.yaml" "${AUTORLBENCH_WORKSPACE:-/workspace}/configs/experiment.yaml"
cat > "${AUTORLBENCH_WORKSPACE:-/workspace}/report.md" <<'REPORT'
# Repair

Disable optimizer offload so Adam moments remain on the GPU, preserving host-memory headroom for native checkpoint serialization under the same 20 GiB container budget. Frozen evidence shows all 10 optimizer updates completed, followed by Ray host-memory eviction during checkpoint saving, with no completed step-10 checkpoint and no kernel OOM kill. The verifier runs fresh training and validates restoration independently.
REPORT
