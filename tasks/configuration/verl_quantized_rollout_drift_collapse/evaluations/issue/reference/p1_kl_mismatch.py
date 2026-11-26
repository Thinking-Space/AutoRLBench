"""P1 · engine-vs-trainer log-prob gap probe.

Quantifies the distribution mismatch between the fp8-quantized rollout engine and
the bf16 trainer, mirroring the training setup exactly:

1. start a vLLM engine with the fp8 blockwise quantization config run_grpo.sh
   injects, then push the HF bf16 weights through the training-identical sync path
   (verl/utils/vllm/vllm_fp8_utils.py: fuse q/k/v and gate/up, blockwise-quantize,
   copy into the live parameters — a standalone engine starts with uncalibrated
   junk until this sync runs);
2. sample responses at the training sampling parameters;
3. rescore each full sequence twice: engine-side per-token log q (vLLM
   prompt_logprobs) vs trainer-side per-token log p (HF bf16 forward);
4. report mean(log q - log p) over response tokens (a sampled estimate of
   KL(q||p)) plus its spread.

--bf16 is the sanity mode (no engine quantization; expect ~0). The fp8 run
measures the weight-quantization component only: the KV cache is not exercised
by this probe.

    python3 p1_kl_mismatch.py --model $WS/models/Qwen2.5-1.5B-Instruct \
        --data <test.parquet> --n-prompts 32 [--bf16]
"""
import argparse
import json

FP8_CFG = {
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(28)],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--bf16", action="store_true", help="sanity: no engine quantization (expect ~0 gap)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).slice(0, args.n_prompts).to_pylist()

    def content(r):
        pr = r["prompt"]
        return pr[0]["content"] if isinstance(pr, list) else json.loads(pr)[0]["content"]

    prompts = [content(r) for r in rows]

    import torch
    from vllm import LLM, SamplingParams
    overrides = None if args.bf16 else {"quantization_config": FP8_CFG}
    engine = LLM(model=args.model, hf_overrides=overrides,
                 gpu_memory_utilization=0.35, max_model_len=2048, enforce_eager=True)
    tok = engine.get_tokenizer()

    if not args.bf16:
        # push the HF bf16 weights through the training-identical sync path
        from verl.utils.vllm.vllm_fp8_utils import quant_weights, _fuse_stacked_weights
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                  device_map="cuda:0")
        HF_W = [(n, p.detach().clone()) for n, p in hf.named_parameters()]

        def sync(m):
            qc = None
            for mod in m.modules():
                qm = getattr(mod, "quant_method", None)
                if qm is not None and getattr(qm, "block_quant", False):
                    qc = qm
                    break
            fused = _fuse_stacked_weights([(n, t.to("cuda:0")) for n, t in HF_W])
            quanted = list(quant_weights(iter(fused), m, qc, dtype=torch.bfloat16))
            params = dict(m.named_parameters())
            wbs = list(getattr(qc, "weight_block_size", (128, 128)))
            scales_transposed = False
            for name, p in params.items():
                if name.endswith((".weight_scale", ".weight_scale_inv")) and p.dim() == 2:
                    base = name[: -len(".weight_scale")] if name.endswith(".weight_scale") \
                        else name[: -len(".weight_scale_inv")]
                    w = params.get(base + ".weight")
                    if w is not None and w.dim() == 2 and w.shape[0] != w.shape[1] and p.shape[0] != p.shape[1]:
                        blocks = ((w.shape[0] + wbs[0] - 1) // wbs[0], (w.shape[1] + wbs[1] - 1) // wbs[1])
                        scales_transposed = tuple(p.shape) == (blocks[1], blocks[0])
                        break
            for name, tensor in quanted:
                p = params.get(name)
                if p is None:
                    continue
                if name.endswith((".weight_scale", ".weight_scale_inv")) and scales_transposed:
                    tensor = tensor.T.contiguous()
                assert p.shape == tensor.shape, (name, tuple(p.shape), tuple(tensor.shape))
                p.data.copy_(tensor.to(p.dtype))
        engine.apply_model(sync)
        print("[sync] engine weights requantized from bf16 (training-identical path)")
        del hf

    # 1) engine samples responses (training sampling params)
    outs = engine.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0,
                                                   max_tokens=args.max_tokens, seed=1))
    texts = [o.prompt + o.outputs[0].text for o in outs]

    # 2) engine-side log q for every token (rescore prompt+response)
    scored = engine.generate(texts, SamplingParams(max_tokens=1, prompt_logprobs=0,
                                                   temperature=0.0))
    seq_logq = []
    for o in scored:
        plp = o.prompt_logprobs  # list, None for first token
        seq_logq.append([float(list(d.values())[0].logprob) for d in plp if d])

    # 3) trainer-side log p for the same tokens (HF bf16 forward)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                              device_map="cuda:0")
    ratios, per_seq = [], []
    for text, logq in zip(texts, seq_logq):
        ids = torch.tensor([tok(text)["input_ids"]], device="cuda:0")
        with torch.no_grad():
            logits = hf(ids).logits.float()
        logp_all = torch.log_softmax(logits[0], dim=-1)
        # logp for token k+1 comes from logits row k; vLLM's prompt_logprobs
        # list (None dropped) indexes the same tokens.
        logp = logp_all[:-1].gather(1, ids[0][1:].unsqueeze(1)).squeeze(1).tolist()
        n = min(len(logq), len(logp))
        raw_gap = [logq[k] - logp[k] for k in range(n)]
        gap = [g for g in raw_gap if g == g and abs(g) != float("inf")]
        ratios += gap
        per_seq.append(round(sum(gap) / len(gap), 4) if gap else float("nan"))
        if not gap and n > 0:
            print(f"[seq] all {n} gaps non-finite")

    ratios = [float(r) for r in ratios]
    if not ratios:
        raise SystemExit("no finite gaps")
    mean_gap = sum(ratios) / len(ratios)
    std_gap = (sum((r - mean_gap) ** 2 for r in ratios) / len(ratios)) ** 0.5
    print("== P1 KL(q||p) sampled estimate over response tokens")
    print(f"sequences: {len(per_seq)}  tokens: {len(ratios)}")
    print(f"mean(logq-logp) = {mean_gap:+.5f}")
    print(f"std  = {std_gap:.5f}   max|gap| = {max(abs(r) for r in ratios):.4f}")
    print("per-seq mean gaps:", json.dumps([round(x, 4) for x in per_seq]))


if __name__ == "__main__":
    main()
