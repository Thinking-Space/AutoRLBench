#!/usr/bin/env python3
"""Verify qat_kv.nvfp4_rollout against a real vLLM NVFP4 engine.

The test exploits a fixed point.  The NVFP4 checkpoint was produced by modelopt
quantizing the BF16 base weights; vLLM loads it and post-processes it into the
FlashInfer/CUTLASS layout.  If our per-step path (scaled_fp4_quant -> swizzle ->
pad) is correct, then re-quantizing *those same BF16 weights* must land back on
what the engine already holds.

So we:
  1. build the engine from the NVFP4 checkpoint,
  2. snapshot layer.weight / layer.weight_scale,
  3. run resync_nvfp4_weights with the BF16 base weights,
  4. compare.

A mismatch is not automatically a bug -- modelopt and vLLM's scaled_fp4_quant
may round E2M1 boundary cases differently -- so this reports the mismatch rate
rather than asserting equality.  A few per-mille of nibbles differing is
rounding; a large fraction means the layout is wrong.

It also checks generation before and after, which is the end-to-end statement:
the engine must still produce the same text.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


PROMPTS = [
    "Let $x$ be a real number such that $x^2 - 5x + 6 = 0$. Find all possible values of $x$.",
    "Compute the remainder when $2^{100}$ is divided by $7$.",
    "What is the sum of the first 20 positive even integers?",
]


def get_model(llm):
    """Reach the nn.Module behind the LLM wrapper.

    Only works under distributed_executor_backend="external_launcher", where
    vLLM v1 runs the engine core in-process.  With the default backend the core
    lives in a separate process and the module is unreachable by attribute
    traversal -- job 22917 hit exactly that.  verl uses external_launcher, so
    this path is also the one the real weight sync takes
    (fsdp_vllm.py:85 reads the same attribute chain).
    """
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", required=True, help="NVFP4 checkpoint dir")
    ap.add_argument("--bf16", required=True, help="BF16 base model dir")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    sys.path.insert(0, "/historical/home/QAT")
    from vllm import LLM, SamplingParams

    from qat_kv.nvfp4_rollout import plan_resync, resync_nvfp4_weights

    # external_launcher expects the caller to own the process group, exactly as
    # verl's torchrun-launched workers do.
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    print(f"[smoke] building engine from {args.nvfp4}", flush=True)
    llm = LLM(
        model=args.nvfp4,
        tensor_parallel_size=1,
        distributed_executor_backend="external_launcher",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        dtype="bfloat16",
        seed=0,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    before_out = [o.outputs[0].text for o in llm.generate(PROMPTS, sp)]

    model = get_model(llm)
    plan = plan_resync(model)
    print(f"[smoke] plan: {len(plan)} NVFP4 layers", flush=True)
    assert len(plan) > 0

    modules = dict(model.named_modules())
    snapshot = {
        name: (
            modules[name].weight.data.clone(),
            modules[name].weight_scale.data.clone(),
        )
        for name in plan
    }

    print(f"[smoke] loading BF16 weights from {args.bf16}", flush=True)
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(args.bf16, dtype=torch.bfloat16)
    weights = {k: v for k, v in hf.state_dict().items()}
    del hf

    print("[smoke] re-quantizing into the live engine", flush=True)
    consumed = resync_nvfp4_weights(model, weights, plan)
    print(f"[smoke] consumed {len(consumed)} source tensors", flush=True)

    # --- tensor-level comparison -------------------------------------------
    worst_w, worst_s, worst_name = 0.0, 0.0, ""
    tot_w = tot_s = 0
    diff_w = diff_s = 0
    for name in plan:
        old_w, old_s = snapshot[name]
        new_w = modules[name].weight.data
        new_s = modules[name].weight_scale.data
        if old_w.shape != new_w.shape:
            print(f"[FAIL] {name}: weight shape {tuple(old_w.shape)} -> {tuple(new_w.shape)}")
            return 1
        if old_s.shape != new_s.shape:
            print(f"[FAIL] {name}: weight_scale shape {tuple(old_s.shape)} -> {tuple(new_s.shape)}")
            return 1
        dw = (old_w != new_w).sum().item()
        ds = (old_s.view(torch.uint8) != new_s.view(torch.uint8)).sum().item()
        tot_w += old_w.numel()
        tot_s += old_s.numel()
        diff_w += dw
        diff_s += ds
        fw = dw / max(1, old_w.numel())
        if fw > worst_w:
            worst_w, worst_s, worst_name = fw, ds / max(1, old_s.numel()), name

    print(
        f"\n[smoke] packed weight bytes differing : {diff_w}/{tot_w} "
        f"({100.0 * diff_w / tot_w:.4f}%)"
    )
    print(
        f"[smoke] block-scale bytes differing   : {diff_s}/{tot_s} "
        f"({100.0 * diff_s / tot_s:.4f}%)"
    )
    print(f"[smoke] worst layer: {worst_name} weight={100*worst_w:.4f}% scale={100*worst_s:.4f}%")

    # --- end-to-end --------------------------------------------------------
    after_out = [o.outputs[0].text for o in llm.generate(PROMPTS, sp)]
    same = sum(1 for a, b in zip(before_out, after_out) if a == b)
    print(f"\n[smoke] greedy generations identical: {same}/{len(PROMPTS)}")
    for i, (a, b) in enumerate(zip(before_out, after_out)):
        if a != b:
            print(f"  --- prompt {i} diverged ---")
            print(f"  before: {a[:200]!r}")
            print(f"  after : {b[:200]!r}")

    ok = same == len(PROMPTS)
    print(f"\n[smoke] {'PASS' if ok else 'GENERATION DIVERGED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
