#!/usr/bin/env python3
"""Post-training-quantize the 1.5B to NVFP4 and export a vLLM-loadable checkpoint.

Two tiers are produced from the same base model:

  W4A16  -- mtq.W4A16_NVFP4_CFG    weights NVFP4, activations BF16
  W4A4   -- mtq.NVFP4_DEFAULT_CFG  weights and activations NVFP4

The KV cache is deliberately *not* quantized here.  vLLM has its own
``--kv-cache-dtype nvfp4`` and that is the path an actual deployment takes, so
the KV tier is applied at eval time instead of being baked into the checkpoint.
(modelopt does offer NVFP4_KV_CFG, targeting ``*[kv]_bmm_quantizer``; using it
would measure modelopt's KV implementation rather than the one we serve with.)

Both configs declare ``algorithm: max`` and block scales of type ``dynamic``.
Dynamic means the per-16-value FP8 scale is computed at runtime, so that level
needs no data -- but the per-tensor amax that ``max`` calibrates does, and it is
cheap, so calibration is run for both tiers rather than reasoned about.

Calibration draws from the cached teacher trajectories (prompt + response token
ids), which is the closest available match to what the model actually sees at
eval time.  NVIDIA calibrates NVFP4 with ~1000 samples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


CONFIGS = {"w4a16": "W4A16_NVFP4_CFG", "w4a4": "NVFP4_DEFAULT_CFG"}


def calibration_batches(args, tokenizer):
    """Yield token-id batches from the cached teacher rollouts."""
    import pyarrow.parquet as pq

    table = pq.read_table(args.calib_data, columns=["prompt_token_ids", "response_token_ids"])
    cols = table.to_pydict()
    seqs = []
    for p, r in zip(cols["prompt_token_ids"], cols["response_token_ids"]):
        ids = list(p) + list(r)
        if len(ids) < 32:
            continue
        seqs.append(ids[: args.calib_seq_len])
        if len(seqs) >= args.calib_samples:
            break
    if len(seqs) < args.calib_samples:
        raise SystemExit(
            f"only {len(seqs)} usable calibration sequences in {args.calib_data}, "
            f"need {args.calib_samples}"
        )
    print(f"[ptq] calibration: {len(seqs)} sequences, "
          f"max len {max(len(s) for s in seqs)}", flush=True)

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(0, len(seqs), args.calib_batch):
        chunk = seqs[i : i + args.calib_batch]
        width = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, s in enumerate(chunk):
            ids[row, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[row, : len(s)] = 1
        yield ids, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-data", required=True)
    ap.add_argument("--calib-samples", type=int, default=1000)
    ap.add_argument("--calib-seq-len", type=int, default=2048)
    ap.add_argument("--calib-batch", type=int, default=2)
    args = ap.parse_args()

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_name = CONFIGS[args.tier]
    cfg = getattr(mtq, cfg_name)
    print(f"[ptq] tier={args.tier} config={cfg_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    batches = list(calibration_batches(args, tokenizer))

    def forward_loop(m):
        with torch.no_grad():
            for n, (ids, mask) in enumerate(batches):
                m(input_ids=ids.cuda(), attention_mask=mask.cuda())
                if (n + 1) % 50 == 0:
                    print(f"[ptq]   calib {n + 1}/{len(batches)} batches", flush=True)

    started = time.time()
    model = mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"[ptq] quantize done in {time.time() - started:.0f}s", flush=True)
    mtq.print_quant_summary(model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Save the modelopt state BEFORE export_hf_checkpoint, which rewrites the
    # module tree into the real-quantized export format.  The state carries the
    # quantizer topology and the calibrated per-tensor amax; QAT restores it
    # onto a fresh BF16 model so training never re-runs calibration (and so
    # every arm shares one frozen amax, which is what makes the arms
    # comparable).
    # modelopt_state() carries the quantizer topology and mode config only --
    # the calibrated per-tensor amax lives in the model's buffers, so save
    # those too.  Restore is: restore_from_modelopt_state() to rebuild the
    # TensorQuantizer tree, then load_state_dict(strict=False) for the amax.
    state_path = out / "modelopt_state.pt"
    buffers = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if "_amax" in k or "_pre_quant_scale" in k
    }
    torch.save(
        {"modelopt_state": mto.modelopt_state(model), "quantizer_buffers": buffers},
        state_path,
    )
    print(
        f"[ptq] wrote modelopt state -> {state_path} "
        f"({len(buffers)} quantizer buffers)",
        flush=True,
    )

    export_hf_checkpoint(model, export_dir=str(out))
    tokenizer.save_pretrained(out)

    cfg_json = out / "hf_quant_config.json"
    meta = {
        "tier": args.tier,
        "modelopt_config": cfg_name,
        "base_model": args.model,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "calib_data": args.calib_data,
        "kv_cache": "not quantized here; applied at eval via vLLM --kv-cache-dtype",
        "exported_quant_config_present": cfg_json.exists(),
    }
    (out / "ptq_meta.json").write_text(json.dumps(meta, indent=2))
    print("[ptq]", json.dumps(meta), flush=True)

    files = sorted(p.name for p in out.iterdir())
    print(f"[ptq] wrote {len(files)} files to {out}: {files[:10]}", flush=True)
    if not any(f.endswith(".safetensors") for f in files):
        raise SystemExit("export produced no safetensors shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
