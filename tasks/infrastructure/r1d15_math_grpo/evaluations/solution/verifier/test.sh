#!/usr/bin/env bash
# Verifier for the "solution" evaluation of the r1d15 math GRPO experiment.
#
# The harness verifier stage (or the BATCH replay entry) has already restored
# the experiment's launch and instrumentation pipeline and run the bounded
# scoring replay (see resources/replay.sh); this script only grades the fresh
# run's records. All checks are deterministic and structural -- there is no
# judge model.
#
# Env:
#   AUTORLBENCH_WORKSPACE  workspace under test   (default /workspace)
#   AUTORLBENCH_RESULTS    where to write results (default /results)
set -uo pipefail

# This script's own directory (the verifier root), so the grader runs the
# verifier's own copy with the verifier's own resources.
verifier="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$verifier/tests/grade_submission.py" \
  --workspace "${AUTORLBENCH_WORKSPACE:-/workspace}" \
  --results "${AUTORLBENCH_RESULTS:-/results}" "$@"
