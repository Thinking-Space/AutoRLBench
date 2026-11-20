#!/bin/bash
set -euo pipefail

# Launcher for an agentic RL run on a mixed query pool.
#
# Everything before the final exec is the pool gate: it pins the three input files by digest,
# validates the pool against its manifest and the unusable-instance registry, derives the run
# identity from the manifest, and sets the rollout batch configuration. The trainer is started
# only if the gate is satisfied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORK_DIR}"

export POOL_DIR="${POOL_DIR:?set POOL_DIR to the pool directory}"
export TRAIN_FILE="${TRAIN_FILE:-${POOL_DIR}/queries.jsonl}"
export POOL_MANIFEST="${POOL_MANIFEST:-${POOL_DIR}/manifest.json}"
export POOL_REGISTRY="${POOL_REGISTRY:-${WORK_DIR}/registry/missing_images.json}"

: "${POOL_SHA256:?Set POOL_SHA256 for the query pool}"
: "${POOL_MANIFEST_SHA256:?Set POOL_MANIFEST_SHA256 for the manifest}"
: "${POOL_REGISTRY_SHA256:?Set POOL_REGISTRY_SHA256 for the registry}"

verify_sha256() {
   local path="$1"
   local expected="$2"
   local label="$3"
   if [ ! -s "${path}" ]; then
      echo "${label} is missing or empty: ${path}" >&2
      return 1
   fi
   local actual
   actual="$(sha256sum "${path}")"
   actual="${actual%% *}"
   if [ "${actual}" != "${expected}" ]; then
      echo "${label} SHA256 mismatch: actual=${actual} expected=${expected} path=${path}" >&2
      return 1
   fi
}

verify_sha256 "${TRAIN_FILE}" "${POOL_SHA256}" "query pool"
verify_sha256 "${POOL_MANIFEST}" "${POOL_MANIFEST_SHA256}" "pool manifest"
verify_sha256 "${POOL_REGISTRY}" "${POOL_REGISTRY_SHA256}" "unusable-instance registry"

python3 - "${TRAIN_FILE}" "${POOL_MANIFEST}" "${POOL_REGISTRY}" <<'PY'
import json
import sys
from collections import Counter

data_path, manifest_path, registry_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
with open(registry_path, encoding="utf-8") as handle:
    registry_payload = json.load(handle)

expected_tools = ("execute_bash", "think", "complete_plan", "finish")
expected_sources = manifest.get("source_counts")
if not isinstance(expected_sources, dict) or not expected_sources:
    raise RuntimeError(f"manifest source_counts must be a non-empty object: {expected_sources!r}")
try:
    expected_sources = {str(source): int(count) for source, count in expected_sources.items()}
except (TypeError, ValueError) as exc:
    raise RuntimeError(f"manifest source_counts must contain integer counts: {expected_sources!r}") from exc
if any(count <= 0 for count in expected_sources.values()):
    raise RuntimeError(f"manifest source_counts must be positive: {expected_sources!r}")
expected_total = int(manifest.get("total_queries", sum(expected_sources.values())))
if expected_total != sum(expected_sources.values()):
    raise RuntimeError(
        f"manifest total_queries does not match source_counts: total={expected_total} "
        f"source_sum={sum(expected_sources.values())}"
    )
excluded_identities = []
for item in registry_payload.get("selected", []):
    source = item.get("source")
    instance_id = item.get("instance_id")
    if source not in expected_sources or not instance_id:
        raise RuntimeError(f"invalid registry entry: {item}")
    excluded_identities.append((source, instance_id))
if len(excluded_identities) != len(set(excluded_identities)):
    raise RuntimeError("duplicate source/instance entries in the registry")
excluded_identities = set(excluded_identities)
expected_excluded_counts = Counter(source for source, _ in excluded_identities)

identities = []
source_counts = Counter()
with open(data_path, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        source = metadata.get("pool_source")
        instance_id = metadata.get("instance_id")
        identities.append((source, instance_id))
        source_counts[source] += 1
        tool_names = tuple(
            tool.get("function", {}).get("name") or tool.get("name")
            for tool in row.get("tools", [])
        )
        if tool_names != expected_tools:
            raise RuntimeError(f"unexpected tools for {(source, instance_id)}: {tool_names}")

if len(identities) != expected_total or len(set(identities)) != expected_total:
    raise RuntimeError(
        f"expected {expected_total} unique queries, got rows={len(identities)} unique={len(set(identities))}"
    )
if dict(source_counts) != expected_sources:
    raise RuntimeError(f"unexpected source counts: {dict(source_counts)}")
if manifest.get("source_counts") != expected_sources:
    raise RuntimeError(f"manifest source counts do not match: {manifest.get('source_counts')}")
present_exclusions = sorted(set(identities) & excluded_identities)
if present_exclusions:
    raise RuntimeError(f"unusable instances remain in the query pool: {present_exclusions}")
manifest_excluded_counts = manifest.get("excluded_instance_counts", {})
if any(
    manifest_excluded_counts.get(source, 0) != expected_excluded_counts.get(source, 0)
    for source in expected_sources
):
    raise RuntimeError(
        "manifest does not match the unusable-instance registry: "
        f"manifest={manifest_excluded_counts} expected={dict(expected_excluded_counts)}"
    )
print(
    "Validated mixed query pool: "
    f"rows={expected_total} sources={dict(source_counts)} excluded_images={len(excluded_identities)}"
)
PY

if [ "${POOL_VALIDATE_ONLY:-0}" = "1" ]; then
   echo "Mixed query pool validate-only check complete."
   exit 0
fi

export POOL_TOTAL="$(python3 -c 'import json,os; print(json.load(open(os.environ["POOL_MANIFEST"]))["total_queries"])')"
export PROJECT_NAME="${PROJECT_NAME:-agentic_rl_mixed_pool}"

if [ -z "${RUN_TAG:-}" ]; then
   _JOB_ID="${JOB_ID:-}"
   if [ -z "${_JOB_ID}" ] || [ "${_JOB_ID}" = "local" ]; then
      _JOB_ID="$(date -u +%Y%m%d_%H%M%S)"
   fi
   export RUN_TAG="mixed${POOL_TOTAL}_s50_rebench35_r2e15_grpo_lr3e6_${_JOB_ID}"
   unset _JOB_ID
fi

export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export ROLLOUT_SHUFFLE=1
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"

echo "Mixed-pool agentic RL:"
echo "  RUN_TAG=${RUN_TAG}"
echo "  TRAIN_FILE=${TRAIN_FILE}"
echo "  NUM_ROLLOUT=${NUM_ROLLOUT} RBS=${ROLLOUT_BATCH_SIZE} OS=${OVER_SAMPLING_BATCH_SIZE} N=${N_SAMPLES_PER_PROMPT}"
echo "  SOURCE_MIX=$(python3 -c 'import json,os; print(json.load(open(os.environ["POOL_MANIFEST"]))["source_counts"])') SHUFFLE_SEED=${ROLLOUT_SEED}"

exec python3 -B "${SCRIPT_DIR}/consume_pool.py"
