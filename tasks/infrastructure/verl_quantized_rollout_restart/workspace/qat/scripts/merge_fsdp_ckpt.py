#!/usr/bin/env python3
"""合并FSDP DTensor checkpoint为普通state_dict（Shard(dim=0) x2 ranks，直接cat local shard）"""

import sys
import torch
from pathlib import Path


def merge_fsdp_state_dict(ckpt_dir, world_size=2):
    ckpt_dir = Path(ckpt_dir)
    rank_sds = []
    for r in range(world_size):
        f = ckpt_dir / f"model_world_size_{world_size}_rank_{r}.pt"
        if not f.exists():
            raise FileNotFoundError(f"缺少rank文件: {f}")
        rank_sds.append(torch.load(f, map_location='cpu', weights_only=False))

    keys = list(rank_sds[0].keys())
    merged = {}
    for k in keys:
        locals_ = [sd[k].to_local() for sd in rank_sds]
        placements = rank_sds[0][k].placements
        if placements != (torch.distributed.tensor.Shard(0),):
            raise ValueError(f"参数 {k} 的placement不是Shard(dim=0): {placements}，需要专门处理")
        merged[k] = torch.cat(locals_, dim=0).clone()

    return merged


def load_model_from_fsdp_ckpt(ckpt_actor_dir, base_model_path, device='cuda:0', dtype=torch.float32):
    from transformers import AutoModelForCausalLM

    merged_sd = merge_fsdp_state_dict(ckpt_actor_dir)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=dtype, trust_remote_code=True,
    )
    missing, unexpected = model.load_state_dict(merged_sd, strict=True)
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    return model.to(device)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-actor-dir', required=True)
    p.add_argument('--base-model', default="/historical/shared/qat/models/Nemotron-Research-Reasoning-Qwen-1.5B-brorl")
    args = p.parse_args()
    m = load_model_from_fsdp_ckpt(args.ckpt_actor_dir, args.base_model)
    print("加载成功，参数量:", sum(p.numel() for p in m.parameters()))
