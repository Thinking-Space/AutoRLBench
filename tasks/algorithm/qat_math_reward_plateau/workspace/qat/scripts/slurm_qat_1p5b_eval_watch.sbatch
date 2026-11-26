#!/usr/bin/env bash
# CPU-only watcher for the explicitly requested 1.5B checkpoint evaluations.
# The evaluator jobs themselves reserve two GPUs per checkpoint.
#SBATCH --job-name=qat1p5b-eval-watch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --output=/historical/shared/qat/r/qat-eval/watch-%j.out
#SBATCH --error=/historical/shared/qat/r/qat-eval/watch-%j.err
#SBATCH --export=ALL
set -euo pipefail
unset LD_PRELOAD
: "${RUN_DIR:?set RUN_DIR}"
: "${ARM:?set ARM}"
exec bash /historical/home/QAT/scripts/watch_qat_1p5b_aime_eval.sh
