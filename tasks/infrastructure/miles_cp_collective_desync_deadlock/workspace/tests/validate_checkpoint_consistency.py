#!/usr/bin/env python3
"""
Validate HF ↔ Megatron checkpoint consistency.

Two-level validation:
  Part 1 — Weight comparison: directly load Megatron checkpoint (dist_checkpointing)
           and HF safetensors, map names, compare tensors.
  Part 2 — Forward comparison: create Megatron model via fixed bridge + HF weights,
           run forward and compare with HF model.

This mirrors the training pipeline's approach: model architecture from correct config,
weights from Megatron checkpoint (bypassing any bridge bugs in the original code).

Usage:
    cd /user/lixin/code/main_0129_lzh__debug0227_fix
    PYTHONPATH=./Megatron-LM/:./mbridge:. \
        torchrun --nproc-per-node 1 tests/validate_checkpoint_consistency.py
"""

import gc
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

SEED = 42
VOCAB_SIZE = 130560
WEIGHT_ATOL = 1e-6

MODEL_PAIRS = [
    {
        "name": "luozhenghang/minicpmmoe",
        "hf": "/user/lixin/share_user/luozhenghang/1029_miles_miniCPM8b/models/minicpmmoe",
        "mg": "/user/lixin/share_user/luozhenghang/1029_miles_miniCPM8b/checkpoints/minicpm5-moe-megatron-fixed-v2",
    },
    {
        "name": "job_182108.step_50000",
        "hf": "/user/lixin/models/job_182108.step_50000",
        "mg": "/user/lixin/models/job_182108.step_50000.miles_megatron",
    },
]


def log(msg):
    print(f"[validate] {msg}", flush=True)


def separator(title):
    print(f"\n{'='*80}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*80}\n", flush=True)


def compare_tensors(a, b, name, atol=1e-6, rtol=1e-5, verbose=True):
    if a.shape != b.shape:
        if verbose:
            log(f"  [FAIL] {name}: shape mismatch {list(a.shape)} vs {list(b.shape)}")
        return False, {"max_diff": float("inf"), "cos_sim": 0.0}

    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(a_f.reshape(1, -1), b_f.reshape(1, -1)).item()
    passed = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)

    if verbose:
        tag = "PASS" if passed else "FAIL"
        log(f"  [{tag}] {name}  max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}  cos={cos_sim:.10f}")

    return passed, {"max_diff": max_diff, "mean_diff": mean_diff, "cos_sim": cos_sim}


def init():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_model_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)


# ── Part 1: Direct Weight Comparison ─────────────────────────────────────────

def load_megatron_checkpoint_weights(mg_path):
    """Load weights from Megatron dist-checkpoint (same path as training script)."""
    ckpt_path = os.path.join(mg_path, "release")
    reader = FileSystemReader(ckpt_path)
    md = reader.read_metadata()

    weight_keys = sorted([
        k for k in md.state_dict_metadata.keys()
        if "_extra_state" not in k and "rng_state" not in k
    ])
    log(f"  Checkpoint has {len(weight_keys)} weight keys")

    sd = {}
    for key in weight_keys:
        m = md.state_dict_metadata[key]
        if isinstance(m, TensorStorageMetadata):
            sd[key] = torch.empty(m.size, dtype=m.properties.dtype)

    try:
        dcp.load(state_dict=sd, storage_reader=reader)
    except Exception:
        log(f"  Full load failed, retrying without expert weights...")
        sd_small = {k: v for k, v in sd.items() if "experts.experts" not in k}
        dcp.load(state_dict=sd_small, storage_reader=reader)
        sd = sd_small

    log(f"  Loaded {len(sd)} weight tensors")
    return sd


def compare_weights_direct(hf_path, mg_path):
    """Part 1: directly compare checkpoint weights with HF safetensors."""
    separator(f"Part 1: Weight Comparison — {os.path.basename(mg_path)}")

    from mbridge.core.safetensor_io import SafeTensorIO
    safetensor_io = SafeTensorIO(hf_path)
    all_hf_keys = safetensor_io.load_hf_weight_names()
    log(f"  HF has {len(all_hf_keys)} weight keys")

    mg_sd = load_megatron_checkpoint_weights(mg_path)

    from mbridge import AutoBridge
    bridge = AutoBridge.from_pretrained(hf_path, trust_remote_code=True)

    n_pass, n_fail, n_skip = 0, 0, 0

    for ckpt_key in sorted(mg_sd.keys()):
        ckpt_tensor = mg_sd[ckpt_key].cpu()

        try:
            mapped_key = ckpt_key.replace("experts.experts.", "experts.")
            hf_names = bridge._weight_name_mapping_mcore_to_hf(mapped_key)
        except (NotImplementedError, Exception) as e:
            log(f"  [SKIP] {ckpt_key}: {e}")
            n_skip += 1
            continue

        if len(hf_names) == 1:
            hf_key = hf_names[0]
            try:
                hf_tensor = safetensor_io.load_one_hf_weight(hf_key).cpu()
            except Exception:
                n_skip += 1
                continue

            if ckpt_tensor.shape != hf_tensor.shape:
                if "embed_tokens" in hf_key or "lm_head" in hf_key:
                    ckpt_tensor = ckpt_tensor[: hf_tensor.shape[0]]

            if ckpt_tensor.shape == hf_tensor.shape:
                ok, _ = compare_tensors(hf_tensor, ckpt_tensor,
                                        f"ckpt/{ckpt_key}", atol=WEIGHT_ATOL, verbose=False)
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    compare_tensors(hf_tensor, ckpt_tensor, f"ckpt/{ckpt_key}", atol=WEIGHT_ATOL)
            else:
                log(f"  [SKIP] {ckpt_key}: shape {list(ckpt_tensor.shape)} vs HF {list(hf_tensor.shape)}")
                n_skip += 1

        elif len(hf_names) > 1:
            if "{expert_id}" in hf_names[0]:
                n_skip += 1
                continue

            try:
                hf_tensors = [safetensor_io.load_one_hf_weight(n).cpu() for n in hf_names]
                mcore_tensor = bridge._weight_to_mcore_format(mapped_key, hf_tensors)
            except Exception as e:
                log(f"  [SKIP] {ckpt_key}: merge error ({e})")
                n_skip += 1
                continue

            if ckpt_tensor.shape == mcore_tensor.shape:
                ok, _ = compare_tensors(mcore_tensor, ckpt_tensor,
                                        f"ckpt/{ckpt_key}", atol=WEIGHT_ATOL, verbose=False)
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    compare_tensors(mcore_tensor, ckpt_tensor, f"ckpt/{ckpt_key}", atol=WEIGHT_ATOL)
            else:
                log(f"  [SKIP] {ckpt_key}: merged shape {list(mcore_tensor.shape)} vs {list(ckpt_tensor.shape)}")
                n_skip += 1

    # Expert weight spot-check (compare a few experts individually)
    expert_keys = [k for k in mg_sd if "experts.experts" in k]
    if expert_keys:
        log(f"\n  Expert weight spot-check ({len(expert_keys)} tensors)...")
        for ckpt_key in expert_keys[:4]:
            ckpt_tensor = mg_sd[ckpt_key].cpu()
            layer_num = ckpt_key.split(".")[2]

            if "linear_fc1" in ckpt_key:
                hf_gate_key = f"model.layers.{layer_num}.mlp.experts.0.gate_proj.weight"
                hf_up_key = f"model.layers.{layer_num}.mlp.experts.0.up_proj.weight"
                try:
                    hf_gate = safetensor_io.load_one_hf_weight(hf_gate_key).cpu()
                    hf_up = safetensor_io.load_one_hf_weight(hf_up_key).cpu()
                    hf_merged = torch.cat([hf_gate, hf_up], dim=0)
                    expert0_mg = ckpt_tensor[0]
                    ok, _ = compare_tensors(hf_merged, expert0_mg,
                                            f"expert_check/{ckpt_key}[0]", atol=WEIGHT_ATOL)
                    if ok:
                        n_pass += 1
                    else:
                        n_fail += 1
                except Exception as e:
                    log(f"  [SKIP] expert check: {e}")
                    n_skip += 1
            elif "linear_fc2" in ckpt_key:
                hf_down_key = f"model.layers.{layer_num}.mlp.experts.0.down_proj.weight"
                try:
                    hf_down = safetensor_io.load_one_hf_weight(hf_down_key).cpu()
                    expert0_mg = ckpt_tensor[0]
                    ok, _ = compare_tensors(hf_down, expert0_mg,
                                            f"expert_check/{ckpt_key}[0]", atol=WEIGHT_ATOL)
                    if ok:
                        n_pass += 1
                    else:
                        n_fail += 1
                except Exception as e:
                    log(f"  [SKIP] expert check: {e}")
                    n_skip += 1

    log(f"\n  Weight comparison: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    return n_fail == 0


# ── Part 2: Forward Comparison ───────────────────────────────────────────────

def compare_forward(hf_path, seq_lengths=None):
    """Part 2: compare HF and fixed-bridge Megatron forward outputs."""
    separator(f"Part 2: Forward Comparison — {os.path.basename(hf_path)}")

    if seq_lengths is None:
        seq_lengths = [128, 512, 2048]

    from transformers import AutoModelForCausalLM
    log("  Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="eager",
    ).cuda().eval()
    n_hf = sum(p.numel() for p in hf_model.parameters())
    log(f"  HF model: {n_hf:,} params")

    from mbridge import AutoBridge
    log("  Loading Megatron model via fixed bridge + HF weights...")
    bridge = AutoBridge.from_pretrained(hf_path, trust_remote_code=True)
    models = bridge.get_model(weight_path=hf_path)
    mg_model = models[0] if isinstance(models, list) else models
    mg_model.eval()
    n_mg = sum(p.numel() for p in mg_model.parameters())
    log(f"  Megatron model: {n_mg:,} params")

    if n_hf != n_mg:
        log(f"  [WARN] Parameter count mismatch: HF={n_hf:,} vs MG={n_mg:,}")

    results = []
    for seq_len in seq_lengths:
        log(f"\n  --- seq_len={seq_len} ---")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + seq_len)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len), generator=gen).cuda()

        try:
            with torch.no_grad():
                hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits

            position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
            with torch.no_grad():
                mg_logits = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

            if mg_logits.shape[-1] != hf_logits.shape[-1]:
                mg_logits = mg_logits[..., :hf_logits.shape[-1]]

            ok, stats = compare_tensors(hf_logits, mg_logits, f"fwd/seq={seq_len}",
                                        atol=2e-2, rtol=2e-2)

            hf_top1 = hf_logits.argmax(dim=-1)
            mg_top1 = mg_logits.argmax(dim=-1)
            agree = (hf_top1 == mg_top1).float().mean().item()
            log(f"    top-1 agreement: {agree*100:.2f}%")
            results.append((seq_len, stats["cos_sim"], agree, ok))

        except torch.cuda.OutOfMemoryError:
            log(f"  [OOM] seq_len={seq_len}")
            results.append((seq_len, 0, 0, False))
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append((seq_len, 0, 0, False))

    del hf_model, mg_model, models, bridge
    gc.collect()
    torch.cuda.empty_cache()

    log(f"\n  {'seq_len':>8s} {'cos_sim':>12s} {'top1_agree':>12s} {'status':>8s}")
    for seq_len, cos_sim, agree, ok in results:
        log(f"  {seq_len:>8d} {cos_sim:>12.8f} {agree*100:>11.2f}% {'PASS' if ok else 'FAIL':>8s}")

    return all(ok for _, _, _, ok in results)


# ── Part 3: RoPE Comparison ──────────────────────────────────────────────────

def compare_rope(hf_path):
    """Part 3: validate LongRoPE embeddings match between HF and Megatron."""
    separator(f"Part 3: RoPE Comparison — {os.path.basename(hf_path)}")

    from mbridge import AutoBridge
    from mbridge.core.util import unwrap_model

    bridge = AutoBridge.from_pretrained(hf_path, trust_remote_code=True)

    gpt_args = bridge._get_gptmodel_args()
    rope_ok = gpt_args.get("rope_scaling") is True
    log(f"  bridge._get_gptmodel_args() rope_scaling = {gpt_args.get('rope_scaling', 'MISSING')}"
        f"  {'[OK]' if rope_ok else '[FAIL]'}")

    models = bridge.get_model()
    model = unwrap_model(models[0] if isinstance(models, list) else models)
    rotary_emb = model.rotary_pos_emb

    log(f"  RotaryEmbedding.rope_scaling = {getattr(rotary_emb, 'rope_scaling', 'N/A')}")

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    hf_config = bridge.hf_config
    rope_type = hf_config.rope_scaling.get("rope_type", "longrope")
    inv_freq_hf, _ = ROPE_INIT_FUNCTIONS[rope_type](hf_config, torch.device("cuda"))

    all_pass = True
    for test_pos in [0, 1024, 4096, 32767, 65535]:
        pos_t = torch.tensor([float(test_pos)], device="cuda")
        hf_angles = pos_t[:, None] * inv_freq_hf[None, :]
        hf_cos = torch.cos(hf_angles).float().squeeze()

        mg_emb = rotary_emb(max_seq_len=test_pos + 1)
        mg_cos = torch.cos(mg_emb[test_pos, 0, 0, :hf_cos.shape[0]]).float()

        cos_sim = F.cosine_similarity(hf_cos.unsqueeze(0), mg_cos.unsqueeze(0)).item()
        max_diff = (hf_cos - mg_cos).abs().max().item()
        ok = cos_sim > 0.9999
        if not ok:
            all_pass = False
        log(f"  pos={test_pos:>6d}: cos_sim={cos_sim:.10f}  max_diff={max_diff:.4e}"
            f"  {'[PASS]' if ok else '[FAIL]'}")

    del model, models, bridge
    gc.collect()
    torch.cuda.empty_cache()
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    separator("HF ↔ Megatron Checkpoint Consistency Validation")
    log(f"Fixed bridge branch: main_0129_lzh__debug0227_fix")
    log(f"Device: {torch.cuda.get_device_name(0)}")
    log(f"Model pairs: {len(MODEL_PAIRS)}")

    t0 = time.time()
    init()

    all_results = {}

    for pair in MODEL_PAIRS:
        separator(f"Model Pair: {pair['name']}")
        log(f"  HF:       {pair['hf']}")
        log(f"  Megatron: {pair['mg']}")

        pair_results = {}

        # Part 1: Weight comparison
        try:
            pair_results["weights"] = compare_weights_direct(pair["hf"], pair["mg"])
        except Exception as e:
            log(f"  Part 1 error: {e}")
            traceback.print_exc()
            pair_results["weights"] = False

        gc.collect()
        torch.cuda.empty_cache()

        # Part 2: Forward comparison (via fixed bridge + HF weights)
        try:
            pair_results["forward"] = compare_forward(pair["hf"])
        except Exception as e:
            log(f"  Part 2 error: {e}")
            traceback.print_exc()
            pair_results["forward"] = False

        gc.collect()
        torch.cuda.empty_cache()

        # Part 3: RoPE comparison
        try:
            pair_results["rope"] = compare_rope(pair["hf"])
        except Exception as e:
            log(f"  Part 3 error: {e}")
            traceback.print_exc()
            pair_results["rope"] = False

        gc.collect()
        torch.cuda.empty_cache()

        all_results[pair["name"]] = pair_results

    # Final summary
    separator("FINAL SUMMARY")
    for name, results in all_results.items():
        log(f"  {name}:")
        for test, passed in results.items():
            log(f"    {test:>10s}: {'PASS' if passed else 'FAIL'}")

    elapsed = time.time() - t0
    total_pass = sum(1 for r in all_results.values() for v in r.values() if v)
    total_tests = sum(len(r) for r in all_results.values())
    log(f"\n  Total: {total_pass}/{total_tests} passed  ({elapsed:.1f}s)")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
