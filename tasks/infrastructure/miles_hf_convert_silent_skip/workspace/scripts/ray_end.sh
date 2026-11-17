#!/usr/bin/env bash
# ray_end.sh — copies Ray spill directory to ceph after training.
# BUG: /tmp/ray/ is absent after SIGKILL or container restart.
# In that case, this script exits 1, causing the HF conversion to be skipped.

set -e

RAY_SPILL_DIR="${RAY_SPILL_DIR:-/tmp/ray}"
CEPH_DEST="${CEPH_DEST:-/beijing_ceph/user/xuxiaoang/scratch/ray_spill}"

if [ ! -d "${RAY_SPILL_DIR}" ]; then
  echo "ray_end.sh: ${RAY_SPILL_DIR} does not exist" >&2
  exit 1
fi

cp -r "${RAY_SPILL_DIR}" "${CEPH_DEST}/" && echo "ray_end.sh: copied ${RAY_SPILL_DIR} to ${CEPH_DEST}"
