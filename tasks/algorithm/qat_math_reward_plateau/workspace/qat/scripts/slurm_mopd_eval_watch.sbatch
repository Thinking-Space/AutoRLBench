#!/usr/bin/env bash
# CPU-only checkpoint watcher.  Keeping it as its own Slurm job prevents the
# training batch shell from reaping it when the trainer is launched with exec.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:0
#SBATCH --time=7-00:00:00
#SBATCH --output=/historical/shared/qat/r/mopd-logs/eval-watch-%j.out
#SBATCH --error=/historical/shared/qat/r/mopd-logs/eval-watch-%j.err
#SBATCH --export=ALL

set -euo pipefail
: "${PROFILE:?set PROFILE}"
: "${RUN_ROOT:?set RUN_ROOT}"
exec bash /historical/home/QAT/scripts/watch_mopd_checkpoints_eval.sh
