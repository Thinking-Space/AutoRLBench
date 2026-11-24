#!/bin/bash
#SBATCH --job-name=memchk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/historical/shared/qat/r/qat-gate/memchk-%j.out
set -euo pipefail
unset LD_PRELOAD
PY=/historical/shared/bitdistiller/envs/vllm-0.19.1-test/bin/python
# 复现训练里的分配器设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'PYEOF'
import torch
GiB = 1024**3
def show(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"{tag:22s} torch_alloc={torch.cuda.memory_allocated()/GiB:7.2f}  "
          f"torch_reserved={torch.cuda.memory_reserved()/GiB:7.2f}  "
          f"driver_used={(total-free)/GiB:7.2f}  driver_total={total/GiB:7.2f}", flush=True)
torch.cuda.init(); show("baseline")
x = torch.empty(int(20*GiB), dtype=torch.uint8, device="cuda"); show("after alloc 20GiB")
y = torch.empty(int(40*GiB), dtype=torch.uint8, device="cuda"); show("after alloc +40GiB")
del x, y; torch.cuda.empty_cache(); show("after free+empty_cache")
PYEOF
