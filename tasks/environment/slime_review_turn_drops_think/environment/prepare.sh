#!/usr/bin/env bash
# slime_review_turn_drops_think: prepare stage.
#
# Runs in a separate prepare-stage container: root, the
# slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12 image, **the only networked
# stage**, and $AUTORLBENCH_WORKSPACE is writable. It fetches the start model at a pinned revision.
# The prompt set ships in workspace/data/math_l5_train.jsonl (built by workspace/scripts/make_data.py
# from the shipped MATH-lighteval train parquet); slime itself is the image's /root/slime (commit
# a3f50097, copied read-only into workspace/slime for reference). run_grpo.sh converts the HF
# checkpoint to Megatron torch_dist on its first launch (needs a GPU, so it is not done here) and
# writes it to $WS/models/Qwen3-1.7B_torch_dist.
#
# Destination is what run_grpo.sh reads: MODEL=$WS/models/Qwen3-1.7B.
set -euo pipefail
: "${AUTORLBENCH_WORKSPACE:?}"
: "${AUTORLBENCH_ENVIRONMENT:?}"
W="$AUTORLBENCH_WORKSPACE"
mkdir -p "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# Qwen/Qwen3-1.7B @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
python3 "$AUTORLBENCH_ENVIRONMENT/fetch_hf.py" \
    --repo Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --dest "$W/models/Qwen3-1.7B"
