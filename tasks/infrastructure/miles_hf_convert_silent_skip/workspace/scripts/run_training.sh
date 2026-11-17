#!/usr/bin/env bash
# Training entry silently skips HF conversion — defective version.

set -ex

# BUG: HF conversion is conditional on ray_end.sh exiting 0.
RC=0

python3 -m miles.train --num-rollout 200 "$@" || RC=$?

if [ ${RC} -eq 0 ]; then
  if bash scripts/ray_end.sh; then
    # BUG: This block is skipped when ray_end.sh exits 1.
    # /tmp/ray/ is absent after SIGKILL; ray_end.sh exits 1; conversion never runs.
    python3 tools/convert_torch_dist_to_hf.py \
      "${MEGATRON_MODEL_PATH}" \
      "${HF_OUTPUT_PATH}" \
      "${HF_MODEL_DIR}"
  else
    echo "ray_end.sh failed (RC=$?), skipping HF conversion"
  fi
fi

exit ${RC}
