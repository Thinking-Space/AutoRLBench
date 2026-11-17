#!/usr/bin/env bash
# Training launch script (defective version)
# This script reproduces the megatron 0.18 + miles_plugins kwarg mismatch.
# The crash occurs at get_model() before any training step.

set -ex

# Reproduce the defect without launching full training
python3 reproduce.py
