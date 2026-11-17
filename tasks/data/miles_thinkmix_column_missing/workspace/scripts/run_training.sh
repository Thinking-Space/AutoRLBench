#!/usr/bin/env bash
# Training launch script (defective version)
# TRAIN_FILE points to a parquet that was never processed by add_thinkmix.py.
# It has no think_mode column, so THINK_MODE=auto silently falls back to full-think.

set -ex

export THINK_MODE=auto
export TRAIN_FILE="${TRAIN_FILE:-data/train.parquet}"

# BUG: TRAIN_FILE has no think_mode column.
# add_thinkmix.py was never run on this parquet.
# miles will log a warning and silently train 100% think samples.

# Verify schema (shows the bug):
python3 -c "
import pyarrow.parquet as pq, sys
f = '${TRAIN_FILE}'
schema = pq.ParquetFile(f).schema_arrow
print('Schema:', schema.names)
if 'think_mode' not in schema.names:
    print('WARNING: think_mode column not found in dataset; falling back to full think mode')
"

echo "Launching training with THINK_MODE=${THINK_MODE}, TRAIN_FILE=${TRAIN_FILE}"
