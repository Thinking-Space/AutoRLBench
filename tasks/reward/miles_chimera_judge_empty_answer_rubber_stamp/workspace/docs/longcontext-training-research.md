# Long-Context RL Training Strategy Research

## Objective
Run MiniCPM5 MoE RL training with rollout-max-response-len=62000 on 8x H100 80GB without OOM, while achieving raw_reward growth.

## Model Architecture
- MiniCPM5 MoE: hidden=1280, 20 layers, 160 experts (top-16), EP=8
- max_position_embeddings: 65536
- Per-GPU memory: ~2 GB model weights (with EP=8), ~10 GB optimizer states

## Problem Analysis

### Original Template Command
The template uses CP=2, max-tokens-per-gpu=16384, rollout-max-response-len=62000. With CP=2, a 63K-token sample becomes ~31.5K per GPU, exceeding the 16384 limit.

### Root Causes of Failure
1. **Truncation + Routing Replay Bug**: truncate_oversized_samples modifies total_lengths metadata but the routing replay tensor retains pre-truncation concatenated size, causing split_with_sizes RuntimeError.
2. **No Activation Recomputation**: Without --recompute-granularity full, all 20 layers activations are stored during backward, consuming extra memory.
3. **No Loss Recomputation**: Output layer logits [T, V] = [16K, 130K] consume ~8.5 GB per microbatch.

## Solution Strategy

Set max_tokens_per_gpu * cp_size >= rollout_max_context_len so no sample exceeds the effective microbatch token budget. This eliminates truncation entirely.

### Working Configuration (Config 2)

| Parameter | Value |
|-----------|-------|
| context-parallel-size | 4 |
| max-tokens-per-gpu | 16384 |
| Effective limit | 65536 (>= max sample 63024) |
| n-samples-per-prompt | 8 |
| global-batch-size | 64 |
| rollout-max-context-len | 65536 |
| recompute-granularity | full (uniform, 1 layer) |
| recompute-loss-function | yes |
| sglang-mem-fraction | 0.55 |
| log-probs-chunk-size | 4096 |

## Experimental Results

### Config 1 (CP=4, n_samples=4, 2 rollouts)

| Metric | Rollout 0 | Rollout 1 |
|--------|-----------|-----------|
| raw_reward | 0.688 | 0.594 |
| response_len (mean) | 20943 | 20699 |
| response_len (max) | 62000 | 62000 |
| truncated_ratio | 9.4% | 9.4% |
| Training peak allocated | 42.16 GB | - |
| Training peak reserved | 61.87 GB | - |

### Config 2 (CP=4, n_samples=8, 3 rollouts)

| Metric | Rollout 0 | Rollout 1 | Rollout 2 |
|--------|-----------|-----------|-----------|
| raw_reward | 0.672 | 0.547 | 0.578 |
| response_len (mean) | 19753 | 23963 | 20366 |
| truncated_ratio | 6.3% | 15.6% | 10.9% |
| Training peak allocated | 42.17 GB | - | - |
| Training peak reserved | 62.45 GB | - | - |

Both configs completed without OOM. Peak memory ~42 GB allocated, ~62 GB reserved. Headroom: ~17 GB on the tightest rank.

## Key Findings

### 1. Training Fits in 80 GB with CP=4 + Recompute
Peak allocated ~42 GB (52% of 80 GB), peak reserved ~62 GB (78%).

### 2. Truncation Bug with Routing Replay
Fixed by skipping truncation when rollout_max_context_len <= effective_limit.

### 3. GRPO Zero-Advantage Problem
rewards ~ 0 and pg_loss ~ 0 even with n_samples=8 because GRPO normalizes within each prompt group. When all samples get the same binary reward, the normalized advantage is 0 and the model receives no gradient signal.

### 4. TorchMemorySaver Incompatibility
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is incompatible with TorchMemorySaver (used for colocated offloading). Must not be set.

## Recommendations for raw_reward Growth
1. Increase n_samples_per_prompt to 16+ for more variance within groups
2. Use harder math problems (target ~50% success rate)
3. Use continuous rewards instead of binary
4. Reduce rollout-max-response-len to prevent 62K cap hits
5. Tune temperature for output diversity

## Usage
```bash
NUM_ROLLOUT=10 bash scripts/community/run-longcontext-research.sh --config 2 \
  --hf-checkpoint /path/to/model \
  --load /path/to/megatron_ckpt \
  --save /path/to/save
```
