#!/usr/bin/env bash
# Verifier for the "solution" evaluation: did the critic's value head actually
# get zero-initialised, and does the training run now report a value loss
# inside the reward scale?
#
# Two independent checks, in increasing cost:
#
#   1. Offline  -- construct the value head and assert it is all zero. Needs
#      only torch. This is the fix itself, checked directly, by
#      tests/check_value_head.py -- the verifier's own copy of the checker,
#      which deliberately does not ship inside the workspace.
#   2. End to end -- assert Stage B's critic warmup loss is below the reward
#      bound in the harness rerun's summary. The harness (run_spec.json) runs
#      the two-stage reproduction before this script; the verifier only reads
#      ${AUTORLBENCH_RESULTS}/experiment_logs/stage_b_value_loss.json. This is
#      what the fix was for, checked by measurement.
#
# Set AUTORLBENCH_SKIP_TRAIN=1 to run only check 1 (useful while iterating, and
# on a machine with no GPU) -- the script says loudly that it did so, and a
# skipped run is not a pass.
#
# Env:
#   WORKSPACE            workspace under test      (default /workspace)
#   AUTORLBENCH_RESULTS     where to write artifacts  (default ${WORKSPACE}/outputs)
#   AUTORLBENCH_VERIFIER    this verifier's directory (default: alongside test.sh)
#   VALUE_LOSS_THRESHOLD reward-scale bound        (default 1.0)
#   AUTORLBENCH_SKIP_TRAIN  1 = offline checks only
set -uo pipefail

# This script's own directory (the verifier root), so the offline check runs
# the verifier's copy -- the checker deliberately does not ship inside the
# workspace, where the issue-level agent could read the diagnosis off it.
verifier="${AUTORLBENCH_VERIFIER:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

WORKSPACE="${WORKSPACE:-/workspace}"
AUTORLBENCH_RESULTS="${AUTORLBENCH_RESULTS:-${WORKSPACE}/outputs}"
THRESHOLD="${VALUE_LOSS_THRESHOLD:-1.0}"

mkdir -p "${AUTORLBENCH_RESULTS}"
fail() { echo "FAIL: $*" >&2; exit 1; }

# ------------------------------------------------------- 1. the fix itself
echo "== check 1/2: value head is zero at construction =="

TARGET="${WORKSPACE}/slime/slime/backends/megatron_utils/model_provider.py"
[ -f "${TARGET}" ] || fail "missing ${TARGET}"

python3 "${verifier}/tests/check_value_head.py" "${TARGET}"
case "$?" in
    0) echo "   value head is zero-initialised" ;;
    1) fail "a freshly constructed value head is not all-zero (see output above)" ;;
    *) fail "check_value_head.py could not run; torch is required" ;;
esac

# The actor must be untouched: LinearForLastLayer is the critic's head, but a
# fix that zeroed every output layer would silently destroy the LM head. Assert
# the actor branch still builds its output layer from the checkpoint.
grep -q "output_size=1" "${TARGET}" \
    || fail "${TARGET} no longer builds a 1-output value head for the critic"

if [ "${AUTORLBENCH_SKIP_TRAIN:-0}" = "1" ]; then
    echo
    echo "AUTORLBENCH_SKIP_TRAIN=1: skipping the end-to-end run."
    echo "SOLUTION_PARTIAL (offline check only; not a pass)"
    exit 2
fi

# --------------------------------------------------- 2. what the fix was for
echo
echo "== check 2/2: Stage B critic warmup loss is inside the reward scale =="

# The harness's system rerun (run_spec.json) executed run.sh with
# OUT_DIR=$AUTORLBENCH_RESULTS/experiment_logs before this script; the fresh
# summary is the only one that counts.
SUMMARY="${AUTORLBENCH_RESULTS}/experiment_logs/stage_b_value_loss.json"

[ -f "${SUMMARY}" ] || fail "no ${SUMMARY}; the system rerun produced no summary"

python3 - "${SUMMARY}" "${THRESHOLD}" <<'PY'
import json, sys

path, threshold = sys.argv[1], float(sys.argv[2])
with open(path) as fh:
    s = json.load(fh)

errors = []

n = s.get("n_critic_steps") or 0
if n < 1:
    errors.append(
        f"no critic optimizer steps in the warmup window (n_critic_steps={n}); "
        "the metric cannot be below the bound by never being reported"
    )

# Rollouts must really have happened -- otherwise a value loss of 0 could just
# mean the critic saw no data.
#
# The evidence is the RolloutManager's per-rollout response-length report, not
# the reward: slime logs `rollout/rewards` from inside `train_actor`, which a
# critic-only warmup stage never calls, so Stage B legitimately has no reward
# metric at all. Response lengths are reported on every rollout regardless.
if (s.get("rollouts_generated") or 0) < 1:
    errors.append("no rollouts were generated; the run did not actually produce episodes")

response_len = s.get("mean_response_len")
if response_len is None or not response_len > 0:
    errors.append(
        "no generated tokens recorded; the critic cannot have been trained on real episodes"
    )

for key in ("first_value_loss", "warmup_value_loss_mean"):
    value = s.get(key)
    if value is None:
        errors.append(f"{key} is missing from the summary")
    elif not value < threshold:
        errors.append(f"{key} = {value:.6f}, expected < {threshold}")

print(f"  critic optimizer steps  {n}")
for key in ("first_value_loss", "warmup_value_loss_mean", "warmup_value_loss_max"):
    value = s.get(key)
    print(f"  {key:<23} {'n/a' if value is None else f'{value:.6f}'}")
print(f"  rollouts generated      {s.get('rollouts_generated')}")
for key, fmt in (("mean_response_len", ".1f"), ("mean_reward", ".4f")):
    value = s.get(key)
    print(f"  {key:<23} {'n/a' if value is None else format(value, fmt)}")

if errors:
    print()
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
PY
[ $? -eq 0 ] || fail "Stage B critic warmup loss is not inside the reward scale"

echo
echo "SOLUTION_VERIFIED"
